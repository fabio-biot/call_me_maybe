*This project has been created as part of the 42 curriculum by fchaput.*

# Call Me Maybe - Introduction to Function Calling in LLMs

## Description

Call Me Maybe translates natural-language prompts into structured function calls.
Instead of answering a request directly, the program selects the best function name and
extracts typed parameters from the prompt.

Example:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": { "a": 2.0, "b": 3.0 }
}
```

The project uses `llm_sdk.Small_LLM_Model` with `Qwen/Qwen3-0.6B`. Function
selection is done with constrained decoding: at each generation step, only tokens that
can still complete a valid function name are allowed.

## Instructions

### Installation

```bash
uv sync
```

or:

```bash
make install
```

### Running the program

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Default paths:

| Argument | Default |
| --- | --- |
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calling_results.json` |

Do not commit `data/output/`; it is generated when the program runs.

### Makefile

| Target | Action |
| --- | --- |
| `install` | Install dependencies with `uv sync` |
| `run` | Run `uv run python -m src $(ARGS)` |
| `debug` | Run the program with `pdb` |
| `clean` | Remove Python caches |
| `lint` | Run `flake8` and `mypy` |

## Algorithm

1. Load the available function definitions from JSON.
2. Encode every function name once with the model tokenizer.
3. Build a prompt containing the user request and the list of available functions.
4. Generate the function name token by token.
5. At each step, compute the valid next token IDs from the encoded function names.
6. Read model logits with `get_logits_from_input_ids`.
7. Pick the highest-logit token only among the valid token IDs.
8. Stop when the generated token sequence exactly matches a function name.
9. Extract parameters from the prompt according to the selected function schema.
10. Write a JSON array containing `prompt`, `name`, and `parameters`.

Invalid function-name tokens are never selected, which keeps the generated function
name constrained to the provided schema.

## Design Decisions

- Function selection uses the LLM logits, not keyword matching.
- Constrained decoding is applied to function names because they are a finite set of
  valid token sequences.
- Parameter extraction is handled separately from function selection to keep the output
  JSON predictable.
- Input and output files are JSON files so they can be validated easily.
- Pydantic models are used to document and validate structured data.
- The implementation only uses public methods from `llm_sdk`.

## Performance Analysis

Expected behavior:

| Goal | Target |
| --- | --- |
| JSON validity | 100% parseable output |
| Function selection | 90%+ on clear prompts |
| Runtime | Under 5 minutes for the provided batch |
| Reliability | Graceful handling of missing or invalid files |

Constrained decoding improves reliability because the model cannot generate a function
name outside the available definitions. The weakest part is parameter extraction, because
natural language can contain ambiguous strings, numbers, or regex instructions.

## Challenges Faced

- Small LLMs are unreliable when asked to freely generate JSON.
- Tokenized function names may share prefixes, so valid next-token filtering must be
  done from token sequences, not raw strings.
- Parameters must match the schema exactly, including argument names and types.
- File paths and generated output must remain compatible with the correction command.

## Testing Strategy

1. Run `make lint` to check `flake8` and `mypy`.
2. Run the program with the default input files.
3. Check that `data/output/function_calling_results.json` is valid JSON.
4. Verify that every object contains exactly `prompt`, `name`, and `parameters`.
5. Compare selected function names and parameter types against
   `functions_definition.json`.
6. Test edge cases: missing files, invalid JSON, empty prompts, decimal numbers, quoted
   strings, and multi-parameter functions.

## Example Usage

```bash
uv run python -m src
```

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Example output:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  }
]
```

## Resources

- [uv documentation](https://docs.astral.sh/uv/)
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- `llm_sdk` package provided with the project
- Project subject: constrained decoding and function calling requirements

### AI Usage

AI was used to help structure the README, summarize the subject requirements, and
clarify wording around constrained decoding, testing, and usage commands. The code,
algorithm, and final behavior must still be reviewed, tested, and understood by the
student before evaluation.
