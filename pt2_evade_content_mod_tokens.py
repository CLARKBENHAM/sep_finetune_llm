# %%
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from itertools import takewhile, accumulate
import time
import ast
from collections import Counter
import glob
import re

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# from pyarrow import parquet as pq
from concurrent.futures import ThreadPoolExecutor
import os
import copy
from openai import OpenAI

from src.make_prompts import *
from src.utils import (
    between_tokens,
    git_hash,
    chat_to_str,
    num_tokens_from_messages,
    num_tokens_from_string,
    MX_TOKENS,
    end_of_convo,
    take_last_tokens,
)

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)
ORD_USE_BETWEEN = [
    0,  # most unique
    8,  # the delete char
    11,  # min matching  re.match('\s', chr(i)) in self_sync_ords
    # also min(works3) i.ie not changing ss1, ss2, h1
    190,  # max(works3)
    192,  # min matching regex.match('[^\r\n\p{L}\p{N}]?\p{L}+' in self_sync_ords
    # also min([i for n,i in num_sep_1_presuf if n == num_sep_1_presuf[1][0]])
    255,  # max([i for n,i in num_sep_1_presuf if n == num_sep_1_presuf[1][0]])
    1000,  # big and round, for randomness. not self_sync nor ascii
]

MN_TOKENS = 50


def chat_max_scores(openai_moderation):
    return max([max(m["category_scores"].values()) for m in openai_moderation])


def chat_max_by_cat(openai_moderation, categories=None):
    """Max score of any chat in convo by category"""
    if categories is None:
        categories = list(openai_moderation[0]["category_scores"].keys())
    return {c: max((r["category_scores"][c] for r in openai_moderation)) for c in categories}


def make_results_frame(
    final_chat_df, ord_vals=ORD_USE_BETWEEN + [None], model="gpt-4-0613", make_new_convo=None
):
    if model not in ("gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4-0613"):
        print(f"WARN: model {model} not expected")
    new_dfs = []
    for ord_val in ord_vals:
        _r_df = pd.DataFrame(index=final_chat_df.index)
        _r_df["new_completion"] = pd.NA
        _r_df["new_oai_mod"] = pd.NA
        _r_df["new_model"] = model
        if ord_val is None:
            _r_df["sent_convo"] = final_chat_df["conversation"].apply(list)
            _r_df["manipulation"] = [{"kind": None, "sep": None}] * len(_r_df["sent_convo"])
            sep = None
        else:
            sep = chr(ord_val)
            # Apply transformations and store results in the new DataFrame
            _r_df["manipulation"] = [{"kind": "between", "sep": sep}] * len(_r_df)
            _r_df["sent_convo"] = final_chat_df["conversation"].apply(
                lambda convo: [{**d, "content": between_tokens(d["content"], sep)} for d in convo]
            )
            if make_new_convo is not None:
                _r_df["sent_convo"] = _r_df.apply(make_new_convo, axis=1)
        new_dfs += [_r_df]
    return pd.concat(new_dfs)


files = [
    "data_dump/lmsys-chat-1m/train-00000-of-00006-4feeb3f83346a0e9.parquet",
    "data_dump/lmsys-chat-1m/train-00001-of-00006-4030672591c2f478.parquet",
    "data_dump/lmsys-chat-1m/train-00002-of-00006-1779b7cec9462180.parquet",
    "data_dump/lmsys-chat-1m/train-00003-of-00006-2fa862bfed56af1f.parquet",
    "data_dump/lmsys-chat-1m/train-00004-of-00006-18f4bdd50c103e71.parquet",
    "data_dump/lmsys-chat-1m/train-00005-of-00006-fe1acc5d10a9f0e2.parquet",
]
chat_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

# sort chat_df by chat_max_scores on 'openai_moderation' column
chat_df = chat_df.sort_values(
    by="openai_moderation", key=lambda s: s.apply(chat_max_scores), ascending=False
)
chat_df = chat_df.reset_index(drop=True)
check_mod_df = pd.concat([
    make_results_frame(chat_df.head(1000), model="text-moderation-stable"),
    make_results_frame(chat_df.head(1000), model="text-moderation-latest"),
])
del check_mod_df["new_completion"]
# %%


def _make_mod_request(i, m):
    e2 = None
    for i in range(5):
        try:
            mod = client.moderations.create(input=i, model=m)
        except Exception as e:
            time.sleep(1.5**i)
            e2 = e
        else:
            return mod.model_dump()  # ["results"]
    print(e2)
    return None


def make_mod_requests(r):
    return [_make_mod_request(i, r["new_model"]) for i in r["sent_convo"]]


with ThreadPoolExecutor(max_workers=5) as executor:
    # print(list(executor.map(make_mod_requests, check_mod_df.head().to_dict("records"))))
    check_mod_df["new_oai_mod"] = list(
        executor.map(make_mod_requests, check_mod_df.to_dict("records"))
    )
check_mod_df.to_pickle(f"data_dump/defaul_mod_comparison_{git_hash()}.pkl")
# %%
