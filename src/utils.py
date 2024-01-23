# %%
import time
import requests

import json
import os
from concurrent.futures import ThreadPoolExecutor
import re
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
import tiktoken
from openai import OpenAI
from copy import deepcopy
from itertools import accumulate

openai_api_key = os.environ["OPENAI_API_KEY"]

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

ses = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=2,
    allowed_methods={"POST"},
)
ses.mount("https://", HTTPAdapter(max_retries=retries))

MX_TOKENS = 4096
SEP = "@"

encoding = tiktoken.encoding_for_model("gpt-4")  # gpt-3.5 to 4 use same encoder: 100k
# tiktoken.encoding_name_for_model("gpt-4") == tiktoken.encoding_name_for_model("gpt-3.5-turbo-instruct")


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


def num_tokens_from_string(string: str, enc=encoding) -> int:
    """Returns the number of tokens in a text string, same formula as above"""
    num_tokens = len(enc.encode(string))
    return num_tokens + 3


def num_tokens_from_messages(
    messages,
    tokens_per_message=3,
    tokens_per_name=1,
    enc=encoding,
):
    """
    theres an overhead of 3 tokens for the overall message
        because every reply is primed with <|start|>assistant<|message|>
    plus 3 tokens for each content/role response
    """
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(enc.encode(value))
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


def between_tokens(s, sep, enc=encoding):
    """
    Returns a new string that will tokenize to as the original would
    but with tokenized(sep) between each old token
    """
    tokens = enc.encode(s)
    sep_token = enc.encode(sep)
    new_tokens = [i for t in tokens for i in (t, *sep_token)]  # ? [: -len(sep_token)]
    return enc.decode(new_tokens)


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


def end_of_convo(convo, max_tokens=MX_TOKENS, enc=encoding):
    """
    Take as many tokens as possible from the 'end' of a chat conversation.
    Make user give first and last message, and total convo length is less than max_tokens.
        TODO: How do convos change if user isn't the first?
    max_tokens should be slightly less than real limit
    """
    # Sending any message, regardless of length has a 3 token cost
    max_tokens -= 3
    max_tokens -= 5  # just for wiggle room
    convo = convo[:-1] if convo[-1]["role"] == "assistant" else convo
    n_toks = [num_tokens_from_messages([c]) - 3 for c in convo]
    sums = list(accumulate(n_toks))
    stop_tokens = sums[-1] - max_tokens
    ix = next((ix for ix, c in enumerate(sums) if c > stop_tokens))
    out = deepcopy(convo[ix:])
    if out[0]["role"] != "user":
        assert len(out) > 1
        out = out[1:]
    used_after_first = sum(n_toks[-len(out[1:]) :]) if len(out) > 1 else 0
    tokens_for_first = (
        max_tokens - used_after_first
    )  # add back 3 from above but still have this limit
    if num_tokens_from_messages(out[:1]) - 3 > tokens_for_first:
        c = out[0]["content"]
        c = enc.decode(enc.encode(c)[-tokens_for_first:])
        out[0]["content"] = c
    return out


def take_last_tokens(c, max_tokens=MX_TOKENS, enc=encoding):
    """
    converts a string to only having at most max_tokens
    useful when insert non-single token characters and need to have long completions
    """
    return enc.decode(enc.encode(c)[-max_tokens:])


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


def get_chat_completion(model, s, sep, client=client, **kwargs):
    if isinstance(s, str):
        messages = [
            {
                "role": "user",
                "content": s,  # f"Continue this story with {sep}:```{s}```", # also makes words 'worse'
            }
        ]
    else:
        messages = s
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stop=["Sorry, ", "I'm sorry", "I apologize"],
        max_tokens=1000,
        **kwargs,
    )
    # print(response)
    out = response.choices[0].message.content.replace(sep, "")
    return get_mod(out)


def chat_to_str(conv):
    """Take a series of chat messages and return single string can send to mod api
    lmsys-chat-1m only moderated the user prompts not the responses
    From: https://github.com/lm-sys/FastChat/blob/722ab0299fd10221fa4686267fe068a688bacd4c/fastchat/serve/monitor/tag_openai_moderation.py#L33
    """
    # return "\n".join([x["content"] for x in conv if x["role"] == "user"])
    return "\n".join([x["content"] for x in conv])
