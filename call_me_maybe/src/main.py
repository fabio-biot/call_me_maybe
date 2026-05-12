from llm_sdk import Small_LLM_Model
import sys
import json


def is_valid_json_prefix(text: str) -> bool:
    stack = 0
    in_string = False
    for i, c in enumerate(text):
        if c == '"' and (i == 0 or text[i-1] != "\\"):
            in_string = not in_string
        if not in_string:
            if c == "{":
                stack += 1
            elif c == "}":
                stack -= 1
                if stack < 0:
                    return False
    return True


def allowed_chars(state):
    if state == "START":
        return set("{")
    if state == "KEY":
        return set('"abcdefghijklmnopqrstuvwxyz_')
    if state == "AFTER_KEY":
        return set(": ")
    if state == "VALUE":
        return set('"abcdefghijklmnopqrstuvwxyz_ ')
    if state == "END":
        return set("}")
    return set()

def next_state(state, char):
    if state == "START":
        if char == "{":
            return "KEY"
    if state == "KEY":
        if char == '"':
            return "AFTER_KEY"
    if state == "AFTER_KEY":
        if char == ":":
            return "VALUE"
    if state == "VALUE":
        if char == '"':
            return "END"
    if state == "END":
        if char == "}":
            return "DONE"
    return None


def select_token(model, tokens, logits, state):
    logits = logits[-1] if isinstance(logits[0], list) else logits
    top = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    for token_id, _ in top[:50]:
        candidate = tokens + [token_id]
        text = model.decode(candidate)
        if len(text) == 0:
            continue
        last_char = text[-1]
        print(state)
        print(text)
        new_state = next_state(state, last_char)
        if new_state is not None:
            return token_id, new_state
    return None, state


def generate_json(model, prompt, max_steps=10):
    tokens = model.encode('{"Question Description":"')[0].tolist()
    state = "KEY"
    for _ in range(max_steps):
        logits = model.get_logits_from_input_ids(tokens)
        token, state = select_token(model, tokens, logits, state)
        if token is None or state == "DONE":
            break
        tokens.append(token)
    return model.decode(tokens)

def read_json():
    i = 0
    functions = []
    with open("data/input/functions_definition.json", "r") as f:
        json_d = json.load(f)
    # print(json_d)
    functions.append(json_d[0]['name'])
    while i != len(json_d):
        # print(json_d[i]['name'])
        functions.append(json_d[i]['name'])
        # (json_d[i]['name'])
        i += 1
    return functions

def encode_functions(availiable_funcs: list[str], model: Small_LLM_Model):
    encoded_funcs = {}
    for name in availiable_funcs:
        encoded_funcs[name] = model.encode(name)[0].tolist()
    return encoded_funcs

def score_function(model, prompt_tokens, func_tokens):
    tokens = prompt_tokens.copy()
    score = 0.0

    for t in func_tokens:
        logits = model.get_logits_from_input_ids(tokens)
        last_logits = logits[-1]

        score += float(last_logits[t])

        tokens.append(t)

    return score / len(func_tokens)

def select_best_function(prompt_tokens, func_map, model):
    best_score = -1e9
    best_func = None

    for name, tokens in func_map.items():
        score = score_function(model, prompt_tokens, tokens)
        print(f"{score} - ")
        logits = model.get_logits_from_input_ids(tokens)
        print("LOGITS TYPE:", type(logits))
        print("LOGITS VALUE:", logits[:10])
        if best_func is None or score > best_score:
            best_score = score
            best_func = name

    return best_func

basejson  = {
  "prompt": "...",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
def complete_json(base: dict, prompt: str):
    base['prompt'] = prompt
    print(base)

def is_complete_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except:
        return False


def main():
    print("Hello from call-me-maybe!")
    model = Small_LLM_Model()
    functions = read_json()
    prompt = sys.argv[1]
    context = f"""
    User request:
    {prompt}
    Available functions:
    """
    for func in functions:
        context += f"- {func}\n"
    context += "\nBest function:\n"

    if len(sys.argv) < 2:
        print("Usage: uv run python -m main \"<prompt>\"")
        return

    func_map = encode_functions(functions, model)
    prompt_tokens = model.encode(context)[0].tolist()
    best_func = select_best_function(prompt_tokens, func_map, model)
    result = {
        "prompt": prompt,
        "name": best_func,
        "parameters": {}
    }

    print(json.dumps(result, indent=2))

    
if __name__ == "__main__":
    main()