# %% Testing Multiple models
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind_from_stats, chisquare, norm
from joblib import Parallel, delayed
from itertools import takewhile, accumulate, combinations
import time
import ast
from collections import Counter
import glob
import re
import gc
import asyncio

# from pyarrow import parquet as pq
from concurrent.futures import ThreadPoolExecutor
import os
import copy
from openai import OpenAI

from src.make_prompts import *
from src.utils import (
    between_tokens,
    get_oa_mod,
    chat_to_str,
    num_tokens_from_messages,
    num_tokens_from_string,
    MX_TOKENS,
    end_of_convo,
    take_last_tokens,
    git_hash,
    get_enc,
)
from src.llm_requests import BatchRequests
from src.multi_model_tokenizers import (
    AnthropicTokenizer,
    GemmaTokenizer,
    LlamaTokenizer,
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

HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]
# HUGGING_FACE_API = "https://huggingface.co/api/datasets/lmsys/lmsys-chat-1m"
pd.set_option("display.max_colwidth", 1000)

oa_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

final_chat_df = pd.read_pickle("data_dump/final_chat_df_d6767b3.pkl")
final_chat_dfc = pd.read_pickle("data_dump/final_chat_dfc_f1978a7.pkl")


def make_results_frame(
    final_chat_df,
    ord_vals=ORD_USE_BETWEEN + [None],
    model="gpt-4-0613",
    make_new_convo=None,
    enc=None,
):
    if model not in ("gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4-0613") or (
        model in ("claude-3-opus-20240229") and enc != "anthropic"
    ):
        print(f"WARN: model {model} not expected")
    new_dfs = []
    for ord_val in ord_vals:
        _r_df = pd.DataFrame(index=final_chat_df.index)
        _r_df["new_completion"] = pd.NA
        _r_df["new_oai_mod"] = pd.NA
        _r_df["new_model"] = model
        # TODO: handle max_token size in input?
        if ord_val is None:
            _r_df["sent_convo"] = final_chat_df["conversation"].apply(list)
            _r_df["manipulation"] = [{"kind": None, "sep": None}] * len(_r_df["sent_convo"])
            sep = None
        else:
            sep = chr(ord_val)
            # Apply transformations and store results in the new DataFrame
            _r_df["manipulation"] = [{"kind": "between", "sep": sep}] * len(_r_df)
            _r_df["sent_convo"] = final_chat_df["conversation"].apply(
                lambda convo: [
                    {**d, "content": between_tokens(d["content"], sep, enc=enc)} for d in convo
                ]
            )
            if make_new_convo is not None:
                _r_df["sent_convo"] = _r_df.apply(make_new_convo, axis=1)
        new_dfs += [_r_df]
    return pd.concat(new_dfs)


# %%
# experiment1 = ResultsData(og_chat_df_path="data_dump/final_chat_df_d6767b3.pkl", model=["gpt-3.5" ], ord_vals=[8, 192], enc_name='openai')
# experiment1 = ResultsData(og_chat_df_path="data_dump/final_chat_df_d6767b3.pkl", model=["gpt-4"], ord_vals=[8, 192], enc_name='openai')
# experiment1 = ResultsData(og_chat_df_path="data_dump/final_chat_df_d6767b3.pkl", model=['claude-3-haiku-20240307' ], ord_vals=[8, 192], enc_name='anthropic')
# experiment1 = ResultsData(og_chat_df_path="data_dump/final_chat_df_d6767b3.pkl", model=['gemini-pro'], ord_vals=[8, 192], enc_name='gemma')
#


# oa_results_framec = _framec_for_new_model(model="gpt-4-0613", enc="openai")
# assert oa_results_framec.equals(results_framec.query('new_model=="gpt-4-0613"'))
# oa_results_framec.to_pickle(f"data_dump/oa_results_framec_{git_hash()}.pkl")
# oa_results_framec = pd.read_pickle("data_dump/oa_results_framec_c10ac43.pkl")
#
# gemini_results_framec = _framec_for_new_model(model="gemini-pro", enc="gemma7b")
# gemini_results_framec.to_pickle(f"data_dump/gemini/results_framec_{git_hash()}.pkl")
# print(gemini_results_framec)
# gemini_results_framec = pd.read_pickle("data_dump/gemini/results_framec_c10ac43.pkl")
#
# llama_results_framec = _framec_for_new_model(model="llama-70b", enc="llama2")
# llama_results_framec.to_pickle(f"data_dump/llama/results_framec_{git_hash()}.pkl")
# llama_results_framec = pd.read_pickle("data_dump/llama/results_framec_c10ac43.pkl")

# since anthropic made by echoing back results this requires a post filter

an_results_framec = _framec_for_new_model(model="claude-3-sonnet-20240229", enc="anthropic")
an_results_framec.to_pickle(f"data_dump/an_mod/results_framec_{git_hash()}.pkl")
an_results_framec = pd.read_pickle(f"data_dump/an_mod/results_framec_{git_hash()}.pkl")
print(an_results_framec)


# Made a "Fill out the result frames for other models" in chars_tokenized_seperatly.py
gemini_results_framec = pd.read_pickle("data_dump/gemini/results_framec_c10ac43.pkl")
llama_results_framec = pd.read_pickle("data_dump/llama/results_framec_c10ac43.pkl")

# not made yet
an_results_framec = pd.read_pickle(f"data_dump/an_mod/results_framec_{git_hash()}.pkl")


# %% #TODO: test anthropic, model refused to "continue" some sentances
# Checking anthropic "tokenizer"
def words_to_counter(r, split_chars=f" {chr(8)}{chr(192)}"):
    return Counter(re.split(f"[{split_chars}]", " ".join([d["content"] for d in r[1:]])))


def _calculate_similarity(row):
    an_tokens = words_to_counter(row["sent_convo"])
    final_tokens = words_to_counter(row["conversation"])

    intersection = an_tokens & final_tokens
    return sum(intersection.values()) / sum(an_tokens.values())


# Assuming 'an_results_framec' and 'final_chat_dfc' are the column names
merged_df = pd.merge(an_results_framec, final_chat_dfc, left_index=True, right_index=True)
merged_df = merged_df[["sent_convo", "conversation"]]
merged_df["similarity"] = merged_df.apply(_calculate_similarity, axis=1)

# Check if similarity is greater than or equal to 0.7
merged_df["is_similar"] = merged_df["similarity"] >= 0.7
merged_df["is_similar"]

from itertools import combinations


def count_overlaps(grouped):
    overlaps = {}
    for group1, group2 in combinations(grouped.groups.keys(), 2):
        index1 = set(grouped.get_group(group1).index)
        index2 = set(grouped.get_group(group2).index)
        overlap = len(index1 & index2)
        overlaps[(group1, group2)] = overlap
    return overlaps


filtered_series = an_results_framec["manipulation"].apply(str)[merged_df["is_similar"].values]
grouped = filtered_series.groupby(filtered_series)
grouped.apply(lambda g: set(g.index))
overlaps = count_overlaps(grouped)
overlaps
# why isn't there more overlap? anthropic should've got the same text all 3 times


# %% #### Fill out the result frames for other models, from dfc where was 'worst' per OA mod


final_chat_dfc = pd.read_pickle("data_dump/final_chat_dfc_f1978a7.pkl")
from src.make_prompts import make_prompt13


def prepend_prompt(prompt_fn, sep, convo, role="system"):
    prompt = prompt_fn(sep)
    return [{"content": prompt, "role": role}] + convo


def _framec_for_new_model(model, enc, df, ord_vals):
    return make_results_frame(
        df,
        ord_vals=ord_vals,
        model=model,
        make_new_convo=lambda r: prepend_prompt(
            make_prompt13,
            r["manipulation"]["sep"],
            r["sent_convo"],
            role="system",
        ),
        enc=enc,
    )


# Using only Gemini encoder, want to see if any get fooled
gemini_encodeded = pd.concat(
    [
        _framec_for_new_model(model=m, enc=e, df=final_chat_dfc.iloc[:50], ord_vals=[8])
        for m, e in [
            ("gpt-4-1106-preview", "gemma7b"),
            ("gpt-3.5-turbo-0301", "gemma7b"),
            # cant limit number of output tokens for vertex
            ("gemini-pro", "gemma7b"),
            ("gemini-pro-1.5", "gemma7b"),
            ("chat-bison@002", "gemma7b"),
            ("claude-3-sonnet-20240229", "gemma7b"),
            ("claude-3-haiku-20240307", "gemma7b"),
        ]
    ]
)
gemini_encodeded.to_pickle("data_dump/gemini/results_framec_gemini_encoded.pkl")
gemini_encodeded = pd.read_pickle("data_dump/gemini/results_framec_gemini_encoded.pkl")


# %%
import importlib
import code
import src.llm_requests

importlib.reload(src.llm_requests)
from src.llm_requests import BatchRequests

br = BatchRequests()


async def get_chat_completion(model, messages, sep=None, **kwargs):
    if isinstance(messages, np.ndarray):
        messages = list(messages)  # if np arr then screws up
    response = await br.router_request(model=model, messages=messages, **kwargs)
    if response is None or isinstance(response, Exception):
        return response, None

    out = response.choices[0].message.content
    if sep is not None:
        out = out.replace(sep, "")

    if not out:
        return response, None
    try:
        mod = await br.router_request(input=out, model="text-moderation-stable")
        return out, mod.model_dump()["results"]
    except Exception as e:
        return out, e


async def fill_out_results_async(df_frame, n_loops=1):
    results_df = copy.deepcopy(df_frame)
    missing_ix = results_df["new_oai_mod"].isna()
    while sum(missing_ix) > 0 and n_loops > 0:
        print(f"adding {sum(missing_ix)}")

        # Use the batch_router_requests method from your br object
        responses = asyncio.run(
            asyncio.gather(
                *[
                    get_chat_completion(model, messages, sep)
                    for model, messages, sep in zip(
                        results_df["new_model"][missing_ix],
                        results_df["sent_convo"][missing_ix],
                        results_df["manipulation"][missing_ix].apply(lambda i: i["sep"]),
                    )
                ]
            )
        )

        results_df.loc[missing_ix, "new_completion"] = [comp for (comp, _) in responses]
        results_df.loc[missing_ix, "new_oai_mod"] = [
            (
                m_oai_mod
                if m_oai_mod is None
                or isinstance(m_oai_mod, Exception)
                or isinstance(m_oai_mod, list)
                else [m_oai_mod]
            )
            for (_, m_oai_mod) in responses
        ]

        missing_ix = results_df["new_oai_mod"].isna()
        n_loops -= 1
    return results_df


df = gemini_encodeded.loc[1].copy()
df["sent_convo"] = [
    [
        {
            "role": "user",
            "content": "tell me the shorest happy story you can",
        }
    ]
    for _ in range(7)
]

updated_df = asyncio.run(fill_out_results_async(df, n_loops=1))
updated_df.to_pickle("data_dump/gemini/resultsc_gemini_encoded.pkl")
print(updated_df["new_oai_mod"].isna().sum(), updated_df["new_completion"].isna().sum())
assert updated_df["new_completion"].apply(lambda x: isinstance(x, Exception)).sum() == 0
assert updated_df["new_oai_mod"].apply(lambda x: isinstance(x, Exception)).sum() == 0
assert (
    updated_df["new_completion"].apply(lambda x: isinstance(x, str)).all()
), "Not all entries in 'new_completion' are strings."
# %%
a = asyncio.run(
    br.router_request(
        model="gemini-pro-1.5",
        messages=[
            {
                "role": "user",
                "content": "tell me the shorest happy story you can",
            }
        ],
    )
)
print(a)
