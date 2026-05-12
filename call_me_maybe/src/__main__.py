from llm_sdk import Small_LLM_Model
import sys


def main() -> None:
    model = Small_LLM_Model()

    text = sys.argv[1]
    tokens = model.encode(text)

    print("TOKENS:")
    print(tokens)
    tokens = tokens[0].tolist()
    print(tokens)
    logits = model.get_logits_from_input_ids(tokens)
    top10 = sorted(enumerate(logits),key=lambda x: x[1],reverse=True)[:10]
    print("TOP TOKENS:", top10)
    best_token = top10[0][0]
    print(best_token)
    for i in range(10):
        decoded = model.decode(top10[i][0])
        print("DECODED NEXT TOKEN:", decoded)
    decoded = model.decode([best_token])
    print("DECODED NEXT TOKEN:", decoded)
    for i in range(50):
        tokens = model.encode(text)
        tokens = tokens[0].tolist()
        logits = model.get_logits_from_input_ids(tokens)
        best_token = sorted(enumerate(logits),key=lambda x: x[1],reverse=True)[0][0]
        decoded = model.decode([best_token])
        print(decoded, end="")
        text += decoded

    print("\nDECODED:")
    print(decoded)


if __name__ == "__main__":
    main()