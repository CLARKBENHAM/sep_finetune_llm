import time
import requests

import json
import os
from concurrent.futures import ThreadPoolExecutor
import re
import tiktoken
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
from openai import OpenAI


openai_api_key = os.environ["OPENAI_API_KEY"]

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

MX_TOKENS = 4096
SEP = "@"


def get_mod(s, openai_api_key=openai_api_key):
    data = {"input": s}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_api_key}"}
    url = "https://api.openai.com/v1/moderations"
    response = requests.post(url, headers=headers, data=json.dumps(data))
    try:
        v = response.json()["results"][0]
    except:
        time.sleep(10)
        response = requests.post(url, headers=headers, data=json.dumps(data))
        v = response.json()["results"][0]
    is_flagged = v["flagged"]
    mx = max(v["category_scores"].values())
    return [is_flagged, mx, s]


def num_tokens_from_string(string: str, model="gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string, same formula as above"""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens + 3


def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def text_w_sep(s, sep=SEP):
    # s = anyascii(s)  # utf-8 to nearest ascii, but gpt can handle uft8
    s = re.sub("(“|”|“)", '"', s)
    s = re.sub(r"(\S)", rf"\1{sep}", s)
    # s = re.sub(r"([a-zA-Z0-9])", rf"\1{SEP}", s) # didn't like this
    return s


def balance_text(st, split_on="\n", mx_len=MX_TOKENS // 20):
    "If text too long split into multiple lines, but keep split_on char in pieces"
    if num_tokens_from_string(st) < mx_len:
        return [st]
    s = st.split(split_on)
    chunks = []
    chunk = ""
    for ix, w in enumerate(s):
        if ix < len(s) - 1:
            w += split_on
        if num_tokens_from_string(chunk + w) < mx_len:
            chunk += w
        else:
            if chunk:
                chunks += [chunk]
            chunk = w
    chunks += [chunk]
    if split_on == "\n":
        chunks = [c for i in chunks for c in balance_text(i, split_on=". ")]
    assert st == "".join(chunks), [
        (ix, i, j) for ix, (i, j) in enumerate(zip(st, "".join(chunks))) if i != j
    ]
    return chunks


def get_completion(model, s, sep=SEP, client=client):
    "Get Chat completion"
    s = text_w_sep(s)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Continue this story with {sep}:```{s}```",
            }
        ],
    )
    # print(response)
    out = response.choices[0].message.content.replace(sep, "")
    return (get_mod(out), out)
