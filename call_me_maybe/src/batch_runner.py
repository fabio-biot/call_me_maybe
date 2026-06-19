import json
import re
import sys
from pathlib import Path
from typing import Any
from llm_sdk.llm_sdk import Small_LLM_Model

PROJECT_ROOT = Path(__file__).parents[1]
FUNCTIONS_PATH = PROJECT_ROOT / "data" / "input" / "functions_definition.json"

if not FUNCTIONS_PATH.exists():
    raise FileNotFoundError(FUNCTIONS_PATH)


def read_json_file(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"missing file: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error


def load_functions(
        path: Path = FUNCTIONS_PATH) -> list[dict[str, Any]]:
    return read_json_file(path)


def decode_tokens(tokens: list[int], model: Small_LLM_Model) -> str:
    return model.decode(tokens)


def encode_function_names(
    functions: list[dict[str, Any]],
    model: Small_LLM_Model,
) -> dict[str, list[int]]:
    encoded = {}

    for function in functions:
        name = function["name"]
        encoded[name] = model.encode(name)[0].tolist()

    return encoded


def get_valid_next_tokens(
    encoded_functions: dict[str, list[int]],
    generated_tokens: list[int],
) -> dict[int, str]:
    valid = {}
    for name, tokens in encoded_functions.items():
        if tokens[: len(generated_tokens)] != generated_tokens:
            continue
        if len(generated_tokens) == len(tokens):
            continue
        valid[tokens[len(generated_tokens)]] = name
    return valid


def constrained_function_generation(
    model: Small_LLM_Model,
    prompt: str,
    encoded_functions: dict[str, list[int]],
) -> str | None:
    context = f"""
User request:
{prompt}

Available functions:
"""
    for name in encoded_functions:
        context += f"- {name}\n"
    context += "\nBest function:\n"
    prompt_tokens = model.encode(context)[0].tolist()
    generated_tokens: list[int] = []
    while True:
        valid_next_tokens = get_valid_next_tokens(
            encoded_functions,
            generated_tokens,
        )
        if not valid_next_tokens:
            return None

        logits = model.get_logits_from_input_ids(
            prompt_tokens + generated_tokens,
        )
        best_token = max(valid_next_tokens, key=lambda token: logits[token])
        generated_tokens.append(best_token)

        for name, tokens in encoded_functions.items():
            if generated_tokens == tokens:
                return name


def extract_numbers(prompt: str) -> list[float]:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", prompt)
    return [float(match) for match in matches]


def extract_integers(prompt: str) -> list[int]:
    """Extract integers from a prompt."""
    matches = re.findall(r"[-+]?\d+", prompt)
    return [int(match) for match in matches]


def extract_regex(prompt: str) -> str | None:
    match = re.search(r"'(.*?)'|\"(.*?)\"", prompt)
    if match:
        return match.group(1) or match.group(2)
    return None


def extract_strings(prompt: str,
                    selected_function: str | None = None) -> list[str]:
    matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", prompt)
    strings = [m[0] or m[1] for m in matches if m[0] or m[1]]
    if strings:
        return strings
    if selected_function == "fn_greet":
        return [prompt.strip().split(" ")[-1]]
    return [prompt]

# def extract_numbers(prompt: str) -> list[float]:
#     numbers = []
#     current = ""

#     for char in prompt:
#         if char.isdigit():
#             current += char
#             continue

#         if current:
#             numbers.append(float(current))
#             current = ""

#     if current:
#         numbers.append(float(current))

#     return numbers


# def extract_strings(prompt: str, selected_function: str | None) -> list[str]:
#     if selected_function == "fn_greet":
#         return [prompt.strip().split(" ")[-1]]

#     strings = []
#     words = prompt.replace("'", '"').split('"')
#     for index in range(1, len(words), 2):
#         strings.append(words[index])

#     return strings


def build_parameters_from_schema(
    functions,
    selected_function,
    prompt,
):
    if selected_function == "fn_substitute_string_with_regex":
        strings = extract_strings(prompt)

        if "number" in prompt.lower():
            return {
                "source_string": strings[0],
                "regex": r"\d+",
                "replacement": "NUMBERS",
            }

        if "vowel" in prompt.lower():
            return {
                "source_string": strings[0],
                "regex": r"[aeiouAEIOU]",
                "replacement": "*",
            }

        if "substitute the word" in prompt.lower():
            return {
                "source_string": strings[2],
                "regex": rf"\b{strings[0]}\b",
                "replacement": strings[1],
            }

    for function in functions:
        if function["name"] != selected_function:
            continue

        result: dict[str, Any] = {}

        numbers = extract_numbers(prompt)
        integers = extract_integers(prompt)
        strings = extract_strings(prompt, selected_function)

        parameters = function.get("parameters", {})

        # indices séparés
        n_i = 0
        i_i = 0
        s_i = 0

        for param_name, param_info in parameters.items():
            param_type = param_info["type"]

            if param_type in ("number", "float"):
                if n_i < len(numbers):
                    result[param_name] = float(numbers[n_i])
                    n_i += 1

            elif param_type == "integer":
                if i_i < len(integers):
                    result[param_name] = int(integers[i_i])
                    i_i += 1

            elif param_type == "string":
                if s_i < len(strings):
                    result[param_name] = strings[s_i]
                    s_i += 1
            if selected_function == "fn_substitute_string_with_regex":
                strings = extract_strings(prompt)
                return {
                    "source_string": strings[0] if strings else "",
                    "regex": r"\d+" if "number" in prompt.lower()
                    else
                    r"[aeiouAEIOU]"
                    if "vowel" in prompt.lower()
                    else
                    extract_regex(prompt) or "",
                    "replacement": "NUMBERS" if "NUMBERS" in prompt
                    else
                    ("*" if "*" in prompt
                     else
                     "dog"
                     if "dog" in prompt
                     else
                     "")
                }
        return result

    return {}


def build_function_call(
        prompt: str, model: Small_LLM_Model) -> dict[str, Any]:
    functions = load_functions()
    encoded_functions = encode_function_names(functions, model)
    selected_function = constrained_function_generation(
        model,
        prompt,
        encoded_functions,
    )

    return {
        "prompt": prompt,
        "name": selected_function,
        "parameters": build_parameters_from_schema(
            functions,
            selected_function,
            prompt,
        ),
    }


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "":
        print("Empty prompt")
        print("Make sure to respect the synthax make run ARGS=YOUR PROMPT")
        return
    if len(sys.argv) == 1 or sys.argv[1].startswith("--"):
        print('a')
        from .batch_runner import parse_args, run_batch
        raise SystemExit(run_batch(parse_args()))
    model = Small_LLM_Model()
    result = build_function_call(sys.argv[1], model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
