# %%
import os
from typing import List, Optional

from sentencepiece import SentencePieceProcessor

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# from huggingface_hub import HfApi

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


class LlamaTokenizer:
    def __init__(self, tokenizer_path="data_dump/llama/70b-chat/tokenizer.model", variant="70b"):
        # https://huggingface.co/meta-llama/Llama-2-70b-chat-hf/resolve/main/tokenizer.model?download=true
        self.variant = variant
        if tokenizer_path is None:
            assert False
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


# %%
