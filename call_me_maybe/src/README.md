*This project has been created as part of the 42 curriculum by \<login\>.*

# Call Me Maybe — Introduction to Function Calling in LLMs

## Description

**Call Me Maybe** turns natural-language requests into structured function calls. For *"What is the sum of 2 and 3?"* it returns the function name and typed arguments—not the numeric answer:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": { "a": 2.0, "b": 3.0 }
}
```

The tool uses [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) via `llm_sdk`. **Function selection** uses **constrained decoding** (only token IDs that extend a valid function-name prefix; highest logit among valid IDs). **Parameters** are filled from the prompt per `functions_definition.json`.

## Instructions

### Prerequisites

- Python 3.10+ (3.12 in `.python-version` at repo root)
- [uv](https://docs.astral.sh/uv/)
- Internet on first run (model download)

### Installation

```bash
cd src
uv sync
```

From repo root: `make install`.

### Running the program

```bash
cd src
uv run python -m src \
  --functions_definition ../data/input/functions_definition.json \
  --input ../data/input/function_calling_tests.json \
  --output ../data/output/function_calling_results.json
```

| Argument | Default (repo root) |
|----------|---------------------|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calling_results.json` |

Do not commit `data/output/`.

**Single prompt (dev):**

```bash
uv run python main.py "What is the sum of 2 and 3?"
```

### Makefile

| Target | Action |
|--------|--------|
| `install` | `uv sync` |
| `run` | `uv run python -m src $(ARGS)` |
| `debug` | `uv run python -m pdb -m src` |
| `clean` | Remove `__pycache__` |
| `lint` | `flake8 .` + `mypy .` (subject flags) |

```bash
cd src
uv run flake8 ..
uv run mypy .. --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
```

`uv lint` does not exist—use `make lint` or `uv run flake8` / `uv run mypy`.

## Algorithm: constrained decoding

1. Encode each function `name` once with `model.encode()`.
2. Build context: user request + function list + `Best function:\n` → `prompt_tokens`.
3. Loop: collect next token IDs that extend a valid name prefix; if none, fail. Call `get_logits_from_input_ids(prompt_tokens + generated_tokens)`. Pick **max logit** among allowed IDs only. Append until a full name matches.
4. Extract `parameters` from the prompt per schema (`number`, `string`, …).

Invalid tokens are never chosen (equivalent to logits −∞).

## Design decisions

- Constrain **names** (finite token sequences) before full JSON generation.
- Greedy argmax on valid logits.
- Parameters parsed from NL separately; Pydantic in `models.py`.
- Public `llm_sdk` API only (`encode`, `get_logits_from_input_ids`, …).

## Performance analysis

| Goal | Target | Notes |
|------|--------|-------|
| Valid JSON | 100% | Keys: `prompt`, `name`, `parameters`. |
| Selection | 90%+ | Structural constraint on names. |
| Speed | &lt; 5 min batch | One forward pass per name token. |

Parameter heuristics may fail on edge cases.

## Challenges faced

- Unreliable free-form JSON from small models → prefix-constrained names.
- Shared token prefixes → disambiguation by logits.
- Parameter extraction from free text.
- Use `uv run` for lint if tools are not on PATH.

## Testing strategy

1. Inputs in `data/input/`.
2. Run batch command; check `function_calling_results.json`.
3. Validate keys, types, required parameters.
4. Edge cases and malformed inputs (graceful errors).
5. Lint before submit.

## Example usage

```bash
cd src && uv sync
uv run python -m src \
  --functions_definition ../data/input/functions_definition.json \
  --input ../data/input/function_calling_tests.json \
  --output ../data/output/function_calling_results.json
```

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  }
]
```

```bash
uv run python main.py "Greet john"
```

## Resources

- [uv](https://docs.astral.sh/uv/)
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Transformers generation](https://huggingface.co/docs/transformers/en/generation_strategies)
- [PEP 257](https://peps.python.org/pep-0257/)
- `instructions.txt` (repo root)

### AI usage

AI may help with README structure and tooling explanations. You must understand and defend constrained decoding, parameter logic, and evaluation changes. Replace `\<login\>` before submission.
