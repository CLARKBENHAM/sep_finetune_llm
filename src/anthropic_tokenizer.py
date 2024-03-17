# %%
from anthropic import AsyncAnthropic
import argparse
import asyncio
import json


async def get_tokens(client, to_tokenize: str, model=None) -> None:
    if model is None:
        model = "claude-3-opus-20240229"
    tokens = []
    async with client.messages.stream(
        max_tokens=1000,
        system=(
            "Copy the text between <tocopy> markers. Include trailing spaces or breaklines. Do not"
            " write anything else. One example \nInput: <tocopy>Example sentence.</tocopy>\nOutput:"
            " Example sentence."
        ),
        messages=[
            {
                "role": "user",
                "content": f"<tocopy>{to_tokenize}</tocopy>",
            }
        ],
        model=model,
    ) as stream:
        async for event in stream:
            if event.type == "content_block_delta":
                tokens.append(event.delta.text)
            if event.type == "message_delta":
                total_tokens_usage = event.usage.output_tokens

    return tokens, total_tokens_usage


def tokenize_text(client, to_tokenize: str, model=None) -> None:
    tokens, total_tokens_usage = asyncio.run(get_tokens(client, to_tokenize, model=model))
    return tokens, total_tokens_usage


class AnthropicEncoding:
    """WARN: SLOW!! Makes Requests"""

    def __init__(self, client):
        self.client = client

    def encode(self, s):
        """Returns string chunks, not number of tokens"""
        s_chunks, ntokens = tokenize_text(self.client, s)
        return s_chunks

    def decode(self, s_chunks):
        return "".join(s_chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="The text to tokenize", required=False, default=None)
    parser.add_argument(
        "--file",
        help="A JSONL file with several texts to be tokenized",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--disable-vocab",
        help="Disable vocabulary creation",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    assert args.text or args.file, "You must provide either a text or an input file."

    KEEP_VOCAB = not args.disable_vocab

    # Initialize the Anthropic client. Will use a key exported as ANTHROPIC_API_KEY in your environment.
    client = AsyncAnthropic()

    if args.text:  # Quick execution and print on screen
        tokens, total_tokens_usage = tokenize_text(client, args.text)
        print("Tokens:", tokens)
        print("Number of text tokens:", len(tokens))
        print("Total tokens usage (as of API):", total_tokens_usage)

        with open("anthropic_vocab.jsonl", "a") as f:
            for t in tokens:
                f.write(json.dumps({"token": t}) + "\n")

        if "".join(tokens) != args.text:
            raise Exception(
                """The tokenization resulted in a different string than the original. See below:\n\n========= Original =========\n{}\n\n\n========= Tokenized =========\n{}""".format(
                    args.text, "".join(tokens)
                )
            )

    if args.file:  # Read from file and write to file
        to_tokenize = []

        # Each line is a JSON object that should be appended to to_tokenize
        with open(args.file, "r") as f:
            for line in f:
                to_tokenize.append(json.loads(line))

        for entry in to_tokenize:
            tokens, total_tokens_usage = tokenize_text(client, entry["text"])
            entry["tokens"] = tokens
            entry["number_of_tokens"] = len(tokens)
            entry["api_total_tokens_usage"] = total_tokens_usage
            entry["tokenization_correct"] = "".join(tokens) == entry["text"]

        with open(args.file.replace(".jsonl", "_tokenized.jsonl"), "w") as f:
            for entry in to_tokenize:
                f.write(json.dumps(entry) + "\n")

        with open("anthropic_vocab.jsonl", "a") as f:
            for t in set([t for entry in to_tokenize for t in entry["tokens"]]):
                f.write(json.dumps({"token": t}) + "\n")
