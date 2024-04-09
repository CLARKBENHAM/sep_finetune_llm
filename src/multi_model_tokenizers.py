from anthropic import AsyncAnthropic
import argparse
import asyncio
import json

import os
from typing import List, Optional

from sentencepiece import SentencePieceProcessor

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from llm_requests import BatchRequests

# from huggingface_hub import HfApi
br = BatchRequests()
br.router_rate_limited

# Configure retry strategy
retry_strategy = Retry(
    total=3,  # Total number of retries to allow
    status_forcelist=[429, 500, 502, 503, 504],  # Status codes to retry
    allowed_methods=["HEAD", "GET", "OPTIONS"],  # HTTP methods to retry
    backoff_factor=1,  # Backoff factor for retries
)

# Create a new session with the retry strategy
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]
headers = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}


def download_files(repo_id, filenames, output_dir):
    api = HfApi()
    os.makedirs(output_dir, exist_ok=True)

    for filename in filenames:
        file_info = api.file_metadata(repo_id=repo_id, path=filename)
        download_url = file_info.download_url
        file_path = os.path.join(output_dir, filename)

        response = requests.get(download_url)
        with open(file_path, "wb") as file:
            file.write(response.content)

        print(f"Downloaded {filename} to {file_path}")


class GemmaTokenizer:
    def __init__(self, tokenizer_path="data_dump/gemini/gemma/7b/tokenizer.model", variant="7b"):
        """
        assume gemma similar to Gemini-pro; it's same tokenizer type and train on similar data
        # Assume the tokenizers the same across versions, but haven't tested that
        # https://huggingface.co/google/gemma-7b/tree/main
        """
        self.variant = variant
        if tokenizer_path is None:
            assert variant == "7b"
            repo_id = f"google/gemma-{variant}"
            filenames = [
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
            ]
            output_dir = "data_dump/gemini/gemma/7b"
            download_files(repo_id, filenames, output_dir)

            tokenizer_path = os.path.join(output_dir, "tokenizer.model")

        # Reload tokenizer.
        self.sp_model = SentencePieceProcessor(model_file=tokenizer_path)

        # BOS / EOS token IDs.
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        """Converts a string into a list of tokens."""
        assert isinstance(s, str)
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """Converts a list of tokens into a string."""
        return self.sp_model.decode(t)


class LlamaTokenizer:
    def __init__(self, tokenizer_path="data_dump/llama/70b-chat/tokenizer.model", variant="70b"):
        # https://huggingface.co/meta-llama/Llama-2-70b-chat-hf/resolve/main/tokenizer.model?download=true
        self.variant = variant
        if tokenizer_path is None:
            print("Untested")
            assert variant == "7b"
            repo_id = f"google/gemma-{variant}"
            filenames = [
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
            ]
            output_dir = "data_dump/llama2/"
            download_files(repo_id, filenames, output_dir)

            tokenizer_path = os.path.join(output_dir, "tokenizer.model")

        # Reload tokenizer.
        self.sp_model = SentencePieceProcessor(model_file=tokenizer_path)

        # BOS / EOS token IDs.
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        """Converts a string into a list of tokens."""
        assert isinstance(s, str)
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """Converts a list of tokens into a string."""
        return self.sp_model.decode(t)


##### Anthropic
async def get_tokens(client, to_tokenize: str, model=None, max_retries=5, backoff_factor=2) -> None:
    """
    Model defaults to haiku
    test_tokenization.py showed they're the same, unless models were getting it wrong
    """
    attempt = 0
    while attempt < max_retries:
        try:
            if model is None:
                model = "claude-3-haiku-20240307"
            tokens = []
            async with client.messages.stream(
                max_tokens=1000,
                system=(
                    "Copy the text between <tocopy> markers. Include trailing spaces or"
                    " breaklines. Do not write anything else. One example \nInput:"
                    " <tocopy>Example sentence.</tocopy>\nOutput: Example sentence."
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
        except Exception as e:
            print("ignoring", e)
            attempt += 1
            if attempt >= max_retries:
                raise
            sleep_time = backoff_factor**attempt
            await asyncio.sleep(sleep_time)


def tokenize_text(client, to_tokenize: str, model=None) -> None:
    tokens, total_tokens_usage = asyncio.run(get_tokens(client, to_tokenize, model=model))
    return tokens, total_tokens_usage


class AnthropicTokenizer:
    """WARN: SLOW!! Makes Requests"""

    def __init__(self, client):
        self.client = client

    def encode(self, s):
        """Returns string chunks, not numbered tokens"""
        s_chunks, ntokens = tokenize_text(self.client, s)
        return s_chunks

    def decode(self, s_chunks):
        return "".join(s_chunks)
