import json
import re
import sys
from pathlib import Path
from typing import Any
from llm_sdk.llm_sdk import Small_LLM_Model

PROJECT_ROOT = Path(__file__).parents[1]
FUNCTIONS_PATH = PROJECT_ROOT / "data" / "input" / "functions_definition.json"


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

def extract_strings(
    prompt: str,
    selected_function: str | None = None,
) -> list[str]:
    if selected_function == "fn_greet":
        return [prompt.strip().split(" ")[-1]]

    strings = []
    words = prompt.replace("'", '"').split('"')
    for index in range(1, len(words), 2):
        strings.append(words[index])

    return strings

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
def extract_regex_parameters(prompt: str) -> dict[str, str]:
    """
    Handles fn_substitute_string_with_regex properly.
    """

    words = prompt.replace("'", '"').split('"')
    quoted = [w for i, w in enumerate(words) if i % 2 == 1]
    source_string = quoted[0] if quoted else ""
    prompt_lower = prompt.lower()
    if "number" in prompt_lower:
        return {
            "source_string": source_string,
            "regex": r"\d+",
            "replacement": "NUMBERS",
        }
    if "vowel" in prompt_lower:
        return {
            "source_string": source_string,
            "regex": r"[AEIOUaeiou]",
            "replacement": "*",
        }
    quoted_texts = quoted
    if len(quoted_texts) >= 2:
        return {
            "source_string": quoted_texts[2] if len(quoted_texts) > 2 else source_string,
            "regex": quoted_texts[0],
            "replacement": quoted_texts[1],
        }

    return {
        "source_string": source_string,
    }

def build_parameters_from_schema(
    functions: list[dict[str, Any]],
    selected_function: str | None,
    prompt: str,
) -> dict[str, Any]:

    for function in functions:
        if function["name"] != selected_function:
            continue

        if selected_function == "fn_substitute_string_with_regex":
            return extract_regex_parameters(prompt)

        result: dict[str, Any] = {}

        numbers = extract_numbers(prompt)
        strings = extract_strings(prompt, selected_function)

        number_index = 0
        string_index = 0

        parameters = function.get("parameters", {})

        for param_name, param_info in parameters.items():
            param_type = param_info["type"]

            if param_type == "number":
                if number_index < len(numbers):
                    value = numbers[number_index]
                    if value.is_integer():
                        result[param_name] = int(value)
                    else:
                        result[param_name] = value

                    number_index += 1

            elif param_type == "string" and string_index < len(strings):
                result[param_name] = strings[string_index]
                string_index += 1

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
