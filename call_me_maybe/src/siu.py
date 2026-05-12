from llm_sdk import Small_LLM_Model
import json
import sys


def load_functions(path="data/input/functions_definition.json"):
    with open(path, "r") as f:
        return json.load(f)


def encode_function_names(functions, model):
    encoded = {}

    for func in functions:
        name = func["name"]
        tokens = model.encode(name)[0].tolist()
        encoded[name] = tokens

    return encoded


def get_valid_next_tokens(encoded_funcs, generated_tokens):
    valid = {}

    for name, tokens in encoded_funcs.items():
        if tokens[:len(generated_tokens)] == generated_tokens:
            if len(generated_tokens) == len(tokens):
                continue

            next_token = tokens[len(generated_tokens)]
            valid[next_token] = name

    return valid


def constrained_function_generation(model, prompt, encoded_funcs):

    context = f"""
    User request:
    {prompt}

    Available functions:
    """
    for name in encoded_funcs.keys():
        context += f"- {name}\n"
    context += "\nBest function:\n"
    prompt_tokens = model.encode(context)[0].tolist()
    generated = []
    while True:
        valid_next = get_valid_next_tokens(encoded_funcs, generated)
        if len(valid_next) == 0:
            break
        logits = model.get_logits_from_input_ids(
            prompt_tokens + generated
        )

        best_token = None
        best_score = -1e9

        for token_id in valid_next.keys():

            score = logits[token_id]

            if score > best_score:
                best_score = score
                best_token = token_id

        if best_token is None:
            break

        generated.append(best_token)

        for name, tokens in encoded_funcs.items():
            if generated == tokens:
                return name

    return None


def extract_numbers(prompt):
    numbers = []
    current = ""
    for c in prompt:
        if c.isdigit():
            current += c
        else:
            if current:
                numbers.append(float(current))
                current = ""
    if current:
        numbers.append(float(current))
    return numbers

def extract_strings(prompt, selected_func):
    strings = []
    if selected_func == "fn_greet":
        strings.append(prompt.strip().split(' ')[-1])
    else:
        words = prompt.replace("'", '"').split('"')
        for i in range(1, len(words), 2):
            strings.append(words[i])

    return strings

def build_parameters_from_schema(functions, best_function, prompt):
    for func in functions:
        if func["name"] != best_function:
            continue

        params = func.get("parameters", {})
        result = {}

        numbers = extract_numbers(prompt)
        strings = extract_strings(prompt, best_function)

        num_i = 0
        str_i = 0

        for param_name, param_info in params.items():
            ptype = param_info["type"]

            if ptype == "number":
                if num_i < len(numbers):
                    result[param_name] = numbers[num_i]
                    num_i += 1

            elif ptype == "string":
                if str_i < len(strings):
                    result[param_name] = strings[str_i]
                    str_i += 1

        return result

    return {}

def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print('uv run python main.py "sum of 4 and 5"')
        return

    prompt = sys.argv[1]
    model = Small_LLM_Model()
    functions = load_functions()
    encoded_funcs = encode_function_names(functions, model)
    best_function = constrained_function_generation(model, prompt, encoded_funcs)
    parameters = build_parameters_from_schema(functions, best_function, prompt)
    result = {
        "prompt": prompt,
        "name": best_function,
        "parameters": parameters
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()