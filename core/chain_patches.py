"""
chain.py patches — apply these changes to core/src/one_ai_core/chain.py

There are 3 patches to apply. Each shows the BEFORE and AFTER code.
Use your editor's find-and-replace to locate the exact strings.

PATCH ORDER: Apply 1, then 2, then 3.
"""

# ═══════════════════════════════════════════════════════════════════════
# PATCH 1: Replace _extract_yaml with a more robust version
# ═══════════════════════════════════════════════════════════════════════
#
# The current version doesn't verify that the extracted text parses as a
# YAML dict. When Mistral outputs bare shell commands like
# "sudo apt-get install kubectl", the regex sees a "YAML line" (word
# followed by text) and passes it through. yaml.safe_load then returns
# a string scalar, not a dict.
#
# The new version:
# 1. Tries fence extraction first (same as before)
# 2. Falls back to finding the first line that starts with a known
#    top-level key (metadata:, steps:, error:, version:)
# 3. Validates the result parses as a dict before returning
# 4. If nothing works, returns as-is for the validator to report

PATCH_1_BEFORE = '''
    @staticmethod
    def _extract_yaml(raw: str) -> str:
        """
        Extract clean YAML from an LLM response that may contain:
        - Prose preamble ("Here is the config:\\n\\n```yaml\\n...")
        - Markdown fences (```yaml / ```yml / ``` / ```bash)
        - Trailing commentary after the closing fence

        Strategy:
        1. If a fenced block exists, extract only the content inside it.
        2. If no fence exists, strip any leading prose lines that don't
           look like YAML (i.e. lines that don't start with a key: pattern,
           a list marker, or whitespace).
        3. Return the cleaned string for the validator.
        """
        import re

        # Try to find the first fenced code block of any type
        fence_pattern = re.compile(
            r"```(?:yaml|yml|bash|sh|)\\s*\\n(.*?)\\n?```",
            re.DOTALL | re.IGNORECASE,
        )
        match = fence_pattern.search(raw)
        if match:
            return match.group(1).strip()

        # No fence found — strip leading prose lines.
        # A "YAML line" starts with: a word followed by \':\', \'-\', or whitespace.
        yaml_line = re.compile(r"^(\\s|\\-|\\w[\\w\\s]*:)")
        lines = raw.splitlines()
        for i, line in enumerate(lines):
            if yaml_line.match(line):
                return "\\n".join(lines[i:]).strip()

        # Nothing looks like YAML — return as-is and let the validator report it
        return raw.strip()
'''

PATCH_1_AFTER = '''
    @staticmethod
    def _extract_yaml(raw: str) -> str:
        """
        Extract clean YAML config from an LLM response.

        Handles: prose preambles, markdown fences, trailing commentary,
        shell commands, and Kubernetes manifests mixed with config YAML.
        """
        import re
        import yaml

        # Known top-level keys in our schema (not K8s keys like apiVersion/kind/spec)
        SCHEMA_KEYS = ("metadata:", "steps:", "error:", "version:", "validation:", "rollback:")

        def _is_valid_config(text: str) -> bool:
            """Return True if text parses as a YAML dict with expected keys."""
            try:
                parsed = yaml.safe_load(text)
                if not isinstance(parsed, dict):
                    return False
                # Must contain at least 'metadata' or 'error' (our schema requirement)
                return bool({"metadata", "error"} & set(parsed.keys()))
            except yaml.YAMLError:
                return False

        # Strategy 1: Extract from fenced code block
        fence_pattern = re.compile(
            r"```(?:yaml|yml|bash|sh|)[ \\t]*\\n(.*?)\\n[ \\t]*```",
            re.DOTALL | re.IGNORECASE,
        )
        for match in fence_pattern.finditer(raw):
            candidate = match.group(1).strip()
            if _is_valid_config(candidate):
                return candidate

        # Strategy 2: Find the line where our schema starts (metadata: or error:)
        lines = raw.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(SCHEMA_KEYS):
                candidate = "\\n".join(lines[i:]).strip()
                # Remove any trailing fence or prose
                if "```" in candidate:
                    candidate = candidate[:candidate.index("```")].strip()
                if _is_valid_config(candidate):
                    return candidate

        # Strategy 3: Try the whole thing as-is (maybe it's clean YAML)
        stripped = raw.strip()
        if _is_valid_config(stripped):
            return stripped

        # Nothing parsed as valid config — return as-is for the validator to report
        return stripped
'''


# ═══════════════════════════════════════════════════════════════════════
# PATCH 2: Fix conversation_history double-append in the retry loop
# ═══════════════════════════════════════════════════════════════════════
#
# FIND this block in the run() method (the validation failure handler):

PATCH_2_BEFORE = '''
            if not validation.is_valid:
                error_summary = validation.error_summary()
                logger.warning("Validation failed (attempt %d):\\n%s", attempt, error_summary)
                # Store error for the next retry prompt
                conversation_history.append({"role": "user", "content": error_summary})
                result.error = error_summary
                continue
'''

# REPLACE with:

PATCH_2_AFTER = '''
            if not validation.is_valid:
                error_summary = validation.error_summary()
                logger.warning("Validation failed (attempt %d):\\n%s", attempt, error_summary)
                result.error = error_summary
                # NOTE: Do NOT append to conversation_history here.
                # The retry block at the top of the loop will format the
                # error into RETRY_PROMPT and append that as a single user message.
                continue
'''

# AND FIND the retry block at the top of the for loop:

PATCH_2B_BEFORE = '''
            else:
                # Append retry instruction with validation errors to history
                last_error = conversation_history[-1]["content"]  # set below on failure
                retry_msg = RETRY_PROMPT.format(error_summary=last_error)
                conversation_history.append({"role": "user", "content": retry_msg})
                messages = self._build_messages(request, rag_context, conversation_history)
'''

# REPLACE with:

PATCH_2B_AFTER = '''
            else:
                # Build retry message using the stored validation error
                retry_msg = RETRY_PROMPT.format(error_summary=result.error)
                conversation_history.append({"role": "user", "content": retry_msg})
                messages = self._build_messages(request, rag_context, conversation_history)
'''


# ═══════════════════════════════════════════════════════════════════════
# PATCH 3: Add import yaml at top of chain.py (if not already there)
# ═══════════════════════════════════════════════════════════════════════
#
# The new _extract_yaml uses yaml.safe_load. The import is inside the
# staticmethod so it's self-contained, but if you prefer a top-level
# import, add it near the other imports:
#
#   import yaml
#
# (This is optional — the inline import works fine.)


print("""
APPLY INSTRUCTIONS:
==================

1. Replace core/src/one_ai_core/prompts/__init__.py entirely with the
   new version (prompts__init__.py).

2. In core/src/one_ai_core/chain.py, apply PATCH 1:
   - Find the _extract_yaml method
   - Replace it entirely with the new version

3. In core/src/one_ai_core/chain.py, apply PATCH 2:
   - Find the validation failure block and remove the conversation_history.append
   - Find the retry block and change last_error to result.error

4. After applying, run:
   cd core
   pytest tests/test_core_smoke.py -v          # should still pass (no live LLM)
   pytest tests/ -v --integration              # test with real Ollama
""")
