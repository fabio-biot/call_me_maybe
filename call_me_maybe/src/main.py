import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_sdk import Small_LLM_Model  # noqa: E402

FUNCTIONS_PATH = PROJECT_ROOT / "data" / "input" / "functions_definition.json"


def load_functions(path: Path = FUNCTIONS_PATH) -> list[dict[str, Any]]:
    with path.open("r") as f:
        return json.load(f)


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
    numbers = []
    current = ""

    for char in prompt:
        if char.isdigit():
            current += char
            continue

        if current:
            numbers.append(float(current))
            current = ""

    if current:
        numbers.append(float(current))

    return numbers


def extract_strings(prompt: str, selected_function: str | None) -> list[str]:
    if selected_function == "fn_greet":
        return [prompt.strip().split(" ")[-1]]

    strings = []
    words = prompt.replace("'", '"').split('"')
    for index in range(1, len(words), 2):
        strings.append(words[index])

    return strings


def build_parameters_from_schema(
    functions: list[dict[str, Any]],
    selected_function: str | None,
    prompt: str,
) -> dict[str, float | str]:
    for function in functions:
        if function["name"] != selected_function:
            continue

        result = {}
        numbers = extract_numbers(prompt)
        strings = extract_strings(prompt, selected_function)
        number_index = 0
        string_index = 0

        for parameter_name, parameter_info in function.get("parameters", {}).items():
            parameter_type = parameter_info["type"]

            if parameter_type == "number" and number_index < len(numbers):
                result[parameter_name] = numbers[number_index]
                number_index += 1

            if parameter_type == "string" and string_index < len(strings):
                result[parameter_name] = strings[string_index]
                string_index += 1

        return result

    return {}


def build_function_call(prompt: str, model: Small_LLM_Model) -> dict[str, Any]:
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
    if len(sys.argv) < 2:
        print('Usage: uv run python main.py "sum of 4 and 5"')
        return

    model = Small_LLM_Model()
    result = build_function_call(sys.argv[1], model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
