# one-ai-config

Configuration schema, validator, and code generator for the OpenNebula AI Assistant. This is the contract between the LLM output and the execution layer.

## Pipeline

```
LLM Output (YAML)
      │
      ▼
┌───────────┐
│ Validator  │  Parses YAML, validates against Pydantic schema,
└─────┬─────┘  checks action params, verifies dependency graph
      │
      ▼
┌───────────┐
│ Code Gen   │  Maps each action to Jinja2 templates,
└─────┬─────┘  produces a runnable Python script
      │
      ▼
  deploy.py     Executable script with helm/kubectl/pyone calls,
                pre-checks, post-checks, rollback, and --dry-run
```

## Quick Start

```bash
pip install -e ".[dev]"
```

### Validate LLM output

```python
from one_ai_config import ConfigValidator

validator = ConfigValidator()
result = validator.validate(yaml_string)

if result.is_valid:
    print("Config is valid!")
    print(f"Steps: {len(result.config.steps)}")
else:
    print(result.error_summary())
    # Feed this back to the LLM for retry
```

### Generate executable script

```python
from one_ai_config import ConfigValidator, CodeGenerator

validator = ConfigValidator()
result = validator.validate(yaml_string)

if result.is_valid:
    generator = CodeGenerator()
    script = generator.generate(result.config)

    # Save to file
    script.save("deploy.py")

    # Show what it will do
    print(script.print_summary())

    # Run with: python deploy.py --dry-run
```

### Full pipeline from YAML string

```python
from one_ai_config import parse_config, CodeGenerator

config = parse_config(yaml_string)
script = CodeGenerator().generate(config)
script.save("deploy.py")
```

## Supported Actions

| Category | Actions |
|----------|---------|
| `oneke.namespace` | `create`, `delete`, `list` |
| `oneke.app` | `deploy`, `uninstall`, `upgrade`, `list`, `wait_ready`, `get_status` |
| `oneke.service` | `get_endpoint`, `expose`, `list` |
| `oneke.storage` | `create_pvc`, `list_pvcs`, `delete_pvc` |
| `oneke.cluster` | `get_info`, `get_status`, `list_nodes`, `scale_nodes` |
| `one.vm` | `create`, `delete`, `poweroff`, `resume`, `list` |

## Testing

```bash
pytest tests/ -v
```
