#!/usr/bin/env python3
"""
Patches for chatbot_on — apply after running debug_chain_validator.py

PATCH 1: chain.py — conversation_history double-append bug
PATCH 2: chain.py — _extract_yaml robustness improvements
PATCH 3: finetune test fix (3 failing tests)

Usage:
    # Review this file first, then apply patches manually or use it as a reference.
    # The patches are shown as before/after with context.
"""

# ═══════════════════════════════════════════════════════════════════
# PATCH 1: chain.py — conversation_history double-append
# ═══════════════════════════════════════════════════════════════════
#
# BUG: On validation failure, the code appends the error_summary to
# conversation_history as a "user" message. Then on the next iteration,
# it reads conversation_history[-1] as `last_error`, formats it into
# RETRY_PROMPT, and appends THAT as another "user" message. So the
# history has TWO consecutive user messages per failure, which confuses
# the LLM on subsequent retries.
#
# FIX: Remove the duplicate append. The error_summary should only be
# stored as result.error (for external reporting), not appended to
# conversation_history. The retry prompt itself embeds the error.
#
# LOCATION: core/src/one_ai_core/chain.py, inside the run() method
#
# BEFORE (around the validation failure block):
# ─────────────────────────────────────────────
#
#             if not validation.is_valid:
#                 error_summary = validation.error_summary()
#                 logger.warning("Validation failed (attempt %d):\n%s", attempt, error_summary)
#                 # Store error for the next retry prompt
#                 conversation_history.append({"role": "user", "content": error_summary})
#                 result.error = error_summary
#                 continue
#
# And the retry block at the top of the loop:
#
#             else:
#                 # Append retry instruction with validation errors to history
#                 last_error = conversation_history[-1]["content"]  # set below on failure
#                 retry_msg = RETRY_PROMPT.format(error_summary=last_error)
#                 conversation_history.append({"role": "user", "content": retry_msg})
#                 messages = self._build_messages(request, rag_context, conversation_history)
#
# AFTER:
# ──────
#
#             if not validation.is_valid:
#                 error_summary = validation.error_summary()
#                 logger.warning("Validation failed (attempt %d):\n%s", attempt, error_summary)
#                 # Store error for external reporting and for the next retry prompt
#                 result.error = error_summary
#                 # NOTE: Do NOT append error_summary to conversation_history here.
#                 # The retry block below will format it into RETRY_PROMPT and append that.
#                 continue
#
# And the retry block becomes:
#
#             else:
#                 # Build retry message with the validation errors from the previous attempt
#                 retry_msg = RETRY_PROMPT.format(error_summary=result.error)
#                 conversation_history.append({"role": "user", "content": retry_msg})
#                 messages = self._build_messages(request, rag_context, conversation_history)


# ═══════════════════════════════════════════════════════════════════
# PATCH 2: chain.py — _extract_yaml robustness
# ═══════════════════════════════════════════════════════════════════
#
# The current _extract_yaml uses a regex that may not match all fence
# variants Mistral produces. Also, if the LLM returns ONLY a YAML
# scalar (e.g., a quoted string), _extract_yaml returns it as-is
# and the validator sees a string instead of a dict.
#
# FIX: After extracting, do a quick yaml.safe_load check. If the
# result is not a dict, try harder (look for 'metadata:' or 'steps:'
# as anchor points).
#
# LOCATION: core/src/one_ai_core/chain.py, replace _extract_yaml

IMPROVED_EXTRACT_YAML = '''
    @staticmethod
    def _extract_yaml(raw: str) -> str:
        """
        Extract clean YAML from an LLM response.

        Handles: prose preambles, markdown fences (```yaml/```yml/```),
        trailing commentary, and double-fenced blocks.
        """
        import re
        import yaml

        # Step 1: Try to find a fenced code block
        # Match ```yaml, ```yml, ``` (no language), ```bash, ```sh
        fence_pattern = re.compile(
            r"```(?:yaml|yml|bash|sh)?\\s*\\n(.*?)\\n\\s*```",
            re.DOTALL | re.IGNORECASE,
        )
        match = fence_pattern.search(raw)
        if match:
            candidate = match.group(1).strip()
            # Verify it's a YAML mapping, not a scalar
            try:
                parsed = yaml.safe_load(candidate)
                if isinstance(parsed, dict):
                    return candidate
            except yaml.YAMLError:
                pass
            # If the fenced block didn't parse as a dict, continue to other strategies

        # Step 2: Find the first line that looks like a YAML mapping key
        # (word followed by colon, or list marker)
        yaml_start = re.compile(r"^(\\w[\\w_\\s]*:|\\s*-\\s+\\w)")
        lines = raw.splitlines()
        for i, line in enumerate(lines):
            if yaml_start.match(line):
                candidate = "\\n".join(lines[i:]).strip()
                # Strip any trailing prose after a closing fence
                if "```" in candidate:
                    candidate = candidate[:candidate.index("```")].strip()
                try:
                    parsed = yaml.safe_load(candidate)
                    if isinstance(parsed, dict):
                        return candidate
                except yaml.YAMLError:
                    pass

        # Step 3: Last resort — look for 'metadata:' specifically (required field)
        for i, line in enumerate(lines):
            if line.strip().startswith("metadata:"):
                candidate = "\\n".join(lines[i:]).strip()
                if "```" in candidate:
                    candidate = candidate[:candidate.index("```")].strip()
                return candidate

        # Nothing worked — return stripped raw and let validator report the error
        return raw.strip()
'''

# print(IMPROVED_EXTRACT_YAML)  # Uncomment to see the replacement code


# ═══════════════════════════════════════════════════════════════════
# PATCH 3: finetune tests — 3 failing tests
# ═══════════════════════════════════════════════════════════════════
#
# BUG: Test data in tests/test_finetune_e2e.py has 2-word instructions
# like "Deploy WordPress" and "Deploy app" that fail the
# min_instruction_words=3 filter in DataQualityCleaner.
#
# FIX: Either (a) lengthen the test instructions to 3+ words, or
# (b) pass min_instruction_words=2 in those specific tests.
#
# Option (a) is cleaner — it tests the real production filter:
#
# BEFORE:
#   "Deploy WordPress"    →  2 words, fails min_instruction_words=3
#   "Deploy app"          →  2 words, fails
#
# AFTER:
#   "Deploy WordPress on cluster"     →  4 words, passes
#   "Deploy Redis app now"            →  4 words, passes
#   (or any 3+ word instruction)
#
# Search for these strings in finetune/tests/test_finetune_e2e.py
# and replace with longer versions.

print("Patches documented. Review each section above and apply to the relevant files.")
print()
print("Quick apply steps:")
print("  1. Run debug_chain_validator.py first to confirm the bug")
print("  2. Apply PATCH 1 (conversation_history fix) to core/src/one_ai_core/chain.py")
print("  3. Apply PATCH 2 (_extract_yaml improvement) to core/src/one_ai_core/chain.py")
print("  4. Apply PATCH 3 (lengthen test instructions) to finetune/tests/test_finetune_e2e.py")
print("  5. Run: cd core && pytest tests/test_score_smoke.py -v")
print("  6. Run: cd core && pytest tests/ -v --integration")
print("  7. Run: cd finetune && pytest tests/ -v")
