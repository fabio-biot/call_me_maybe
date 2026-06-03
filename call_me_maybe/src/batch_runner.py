import argparse
import json
import sys
from pathlib import Path
from typing import Any

from llm_sdk import Small_LLM_Model

from .__main__ import (
    build_parameters_from_schema,
    constrained_function_generation,
    encode_function_names,
    read_json_file,
)


PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_FUNCTIONS_PATH = (
    PROJECT_ROOT / "data" / "input" / "functions_definition.json"
)
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "data" / "input" / "function_calling_tests.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "output" / "function_calling_results.json"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the batch runner."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default=str(DEFAULT_FUNCTIONS_PATH),
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
    )
    return parser.parse_args()


def load_functions(path: Path) -> list[dict[str, Any]]:
    """Load and validate function definitions."""
    data = read_json_file(path)
    if not isinstance(data, list):
        raise ValueError("functions_definition must contain a JSON array")

    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each function definition must be an object")
        if not isinstance(item.get("name"), str):
            raise ValueError("each function definition needs a string name")
        if not isinstance(item.get("parameters"), dict):
            raise ValueError("each function definition needs parameters")

    return data


def load_prompts(path: Path) -> list[str]:
    """Load prompt strings from the input JSON file."""
    data = read_json_file(path)
    if not isinstance(data, list):
        raise ValueError("input file must contain a JSON array")

    prompts = []
    for item in data:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("prompt"), str)
        ):
            raise ValueError("each input item must contain a string prompt")
        prompts.append(item["prompt"])

    return prompts


def build_function_call(
    prompt: str,
    functions: list[dict[str, Any]],
    encoded_functions: dict[str, list[int]],
    model: Small_LLM_Model,
) -> dict[str, Any]:
    """Create one function-call object for a prompt."""
    selected_function = constrained_function_generation(
        model,
        prompt,
        encoded_functions,
    )
    if selected_function is None:
        raise ValueError(f"no function selected for prompt: {prompt}")

    return {
        "prompt": prompt,
        "name": selected_function,
        "parameters": build_parameters_from_schema(
            functions,
            selected_function,
            prompt,
        ),
    }


def write_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")


def run_batch(args: argparse.Namespace) -> int:
    try:
        functions_path = Path(args.functions_definition)
        input_path = Path(args.input)
        output_path = Path(args.output)
        functions = load_functions(functions_path)
        prompts = load_prompts(input_path)
        model = Small_LLM_Model()
        encoded_functions = encode_function_names(functions, model)
        results = [
            build_function_call(prompt, functions, encoded_functions, model)
            for prompt in prompts
        ]
        write_results(output_path, results)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {len(results)} result(s) to {output_path}")
    return 0


def main() -> None:
    """Program entrypoint."""
    raise SystemExit(run_batch(parse_args()))


if __name__ == "__main__":
    main()
