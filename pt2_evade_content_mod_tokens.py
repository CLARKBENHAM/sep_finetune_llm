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


# %%
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
chat_df = chat_df.reset_index(drop=True).head(1000)
chat_df.to_pickle(f"data_dump/oai_mod/comparison_base_df{git_hash()}.pkl")


# %%
def make_results_frame(
    final_chat_df, ord_vals=ORD_USE_BETWEEN + [None], model="text-moderation-latest", make_new_convo=None
):
    if model not in ("text-moderation-stable", "text-moderation-latest"):
        print(f"WARN: model {model} not expected")
    new_dfs = []
    for ord_val in ord_vals:
        _r_df = pd.DataFrame(index=final_chat_df.index)
        _r_df["conversation_id"] = final_chat_df["conversation_id"]  # should've added
        _r_df["new_completion"] = pd.NA
        _r_df["new_oai_mod"] = pd.NA
        _r_df["new_model"] = model
        if ord_val is None:
            _r_df["sent_convo"] = final_chat_df["conversation"].apply(list)
            _r_df["manipulation"] = [{"kind": None, "sep": None}] * len(_r_df["sent_convo"])
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


chat_df = pd.read_pickle(f"data_dump/oai_mod/comparison_base_df18bd574.pkl")
check_mod_df = pd.concat([
    make_results_frame(chat_df, model="text-moderation-stable"),
    make_results_frame(chat_df, model="text-moderation-latest"),
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
    if r["new_oai_mod"] is None:
        return [_make_mod_request(c, r["new_model"]) for c in r["sent_convo"]]
    out = r["new_oai_mod"]
    if len(out) != len(r["sent_convo"]):
        print("tossing")
        out = [None] * len(r["sent_convo"])
    # should've sent i['content']? Could just
    out = [
        _make_mod_request(c, r["new_model"]) if o is None else o
        for c, o in zip(r["sent_convo"], out)
    ]
    return out


with ThreadPoolExecutor(max_workers=10) as executor:
    # print(list(executor.map(make_mod_requests, check_mod_df.head().to_dict("records"))))
    check_mod_df["new_oai_mod"] = list(
        executor.map(make_mod_requests, check_mod_df.to_dict("records"))
    )
check_mod_df.to_pickle(f"data_dump/oai_mod/comp_results_{git_hash()}.pkl")
# %%
analysis_mod_df = pd.read_pickle("data_dump/oai_mod/comp_results_18bd574_3.pkl")
d = copy.deepcopy(analysis_mod_df)
assert (analysis_mod_df.sent_convo.apply(len) == analysis_mod_df.new_oai_mod.apply(len)).all()

# explode each turn in 'new_oai_mod' and 'sent_convo' into their own rows
analysis_mod_df["paired"] = analysis_mod_df.apply(
    lambda row: [
        (ix, c, o) for ix, (c, o) in enumerate(zip(row["sent_convo"], row["new_oai_mod"]))
    ],
    axis=1,
)
analysis_mod_df = analysis_mod_df.explode("paired")
analysis_mod_df[["convo_ix", "sent_convo", "new_oai_mod"]] = pd.DataFrame(
    analysis_mod_df["paired"].tolist(), index=analysis_mod_df.index
)
analysis_mod_df = analysis_mod_df.drop(columns=["paired"])

# check exploded correctly
num_made = analysis_mod_df['new_model'].nunique() * analysis_mod_df['manipulation'].apply(str).nunique()
assert d.sent_convo.apply(len).sum() == len(analysis_mod_df)
if 'chat_df' in locals():
    assert chat_df['conversation'].apply(len).sum() * num_made == len(analysis_mod_df)
assert (
    d["new_oai_mod"].apply(lambda l: [d["id"] for d in l]).explode().sort_values().values
    == analysis_mod_df["new_oai_mod"].apply(lambda d: d["id"]).sort_values().values
).all()
assert (
    d.drop_duplicates(subset="conversation_id")
    .set_index("conversation_id")["sent_convo"]
    .apply(len)
    .sort_index()
    == (analysis_mod_df["conversation_id"].value_counts() / num_made).sort_index()
).all()
assert analysis_mod_df.groupby(['conversation_id', 'convo_ix']).ngroups * num_made == len(analysis_mod_df)
assert (d.sent_convo.apply(len).value_counts().sort_index() / num_made == (analysis_mod_df.groupby('conversation_id')['convo_ix'].max()+1).value_counts().sort_index()).all()

#%%
# del d
print(analysis_mod_df.shape)

# cols should be unique on: maybe dup strings; def dup models
uniq_cols = ["mod_model", "convo_str", "man_str"]

# What tried to set content mod model as
analysis_mod_df = analysis_mod_df.rename(columns={"new_model": "_sent_model"})

#analysis_mod_df = analysis_mod_df.query("_sent_model=='text-moderation-latest'")

analysis_mod_df["mod_model"] = analysis_mod_df["new_oai_mod"].apply(lambda d: d["model"])
analysis_mod_df["convo_str"] = analysis_mod_df["sent_convo"].apply(str)
analysis_mod_df["man_str"] = analysis_mod_df["manipulation"].astype(str)

n = len(analysis_mod_df)
analysis_mod_df = analysis_mod_df.drop_duplicates(subset=uniq_cols)
if n - len(analysis_mod_df) > 0:
    print(f"WARN: dropped {n-len(analysis_mod_df)} duplicates based on {uniq_cols}")
# check dropped dups all
check_cols = ["mod_model", "convo_ix", "conversation_id", "man_str"]
assert analysis_mod_df[check_cols].drop_duplicates().shape[0] == analysis_mod_df.shape[0]
d = copy.deepcopy(analysis_mod_df)

#%%
duplicate_mask = d.duplicated(subset=uniq_cols, keep='first')
dropped_rows = analysis_mod_df[duplicate_mask]
print(dropped_rows)
# This makes dups, shouldn't have any dups
# Change this to filter all dups at the start
analysis_mod_df.query("conversation_id=='82af34c6bdee4cf18d5e3b203ce5fefa' and convo_ix==9")['sent_convo']

# %%
analysis_mod_df = copy.deepcopy(d)


# split into x and y values
def explode_moderation_results(df, prefix, keep=None):
    """
    Explode moderation results into separate columns.
    :param df: DataFrame containing the moderation results.
    :param prefix: Prefix for the new columns.
    :return: DataFrame with exploded moderation results.
        drops new_completion and new_oai_mod columns
    """
    exploded_mod = df["new_oai_mod"].apply(lambda d: pd.Series(d["results"][0]["category_scores"]))
    exploded_mod["max_score"] = exploded_mod.apply(max, axis=1)
    exploded_mod["any_flagged"] = df["new_oai_mod"].apply(
        lambda d: int(max(d["results"][0]["categories"].values()))
    )
    exploded_mod = exploded_mod.add_prefix(f"{prefix}_")
    if keep is not None:
        exploded_mod[keep] = df[keep]
    if not exploded_mod.index.is_unique:
        print(
            f"INFO: index non-unique for '{prefix}' {exploded_mod.index.unique()},"
            f" {len(exploded_mod)}"
        )
    return exploded_mod

# already uniq on convo_str so conversation_id and convo_ix are unique here
join_on = ["conversation_id", "convo_ix", "mod_model"]
keep_cols = join_on + ["sent_convo"]
new_only_cols = keep_cols + ["manipulation"]
print(f"Dropping columns: {set(analysis_mod_df.columns) - set(new_only_cols)}")
some_mod = analysis_mod_df["manipulation"].apply(
    lambda d: d["sep"] is not None or d["kind"] is not None
)
new_mod = explode_moderation_results(analysis_mod_df[some_mod], "new", keep=new_only_cols).rename(
    columns={"sent_convo": "new_sent_convo"}
)
default_mod = explode_moderation_results(
    analysis_mod_df[~some_mod], "default", keep=keep_cols
).rename(columns={"sent_convo": "default_sent_convo"})
# %%
merged_df = (
    new_mod.set_index(join_on)
    .merge(default_mod.set_index(join_on), left_index=True, right_index=True, how="left")
    .reset_index()
)
print(merged_df.shape)
# %%
assert (merged_df["new_sent_convo"].apply(type).value_counts() == merged_df["default_sent_convo"].apply(type).value_counts()).all()
for c in join_on:
    assert (
        new_mod[c].value_counts().sort_index() == merged_df[c].value_counts().sort_index()
    ).all(), f"{c}1"
    assert (
        merged_df[c].value_counts().sort_index() + default_mod[c].value_counts().sort_index()
        == analysis_mod_df[c].value_counts().sort_index()
    ).all(), f"{c}2"
assert (
    analysis_mod_df.set_index(join_on)["sent_convo"][some_mod.values]
    .sort_index()
    .equals(merged_df.set_index(join_on)["new_sent_convo"].sort_index())
), "new convo off"
#%%
assert merged_df["new_sent_convo"].apply(lambda d: d['content']).nunique() == len(merged_df)
#%%
# 8203 vs 8200
assert (
    analysis_mod_df.set_index(join_on)["sent_convo"][~some_mod.values].apply(lambda d: d['content'])
    .sort_index()
    .equals(merged_df.set_index(join_on)["default_sent_convo"].sort_index().apply(lambda d: d['content']).unique())
), "default convo off"

#%%
assert (
    analysis_mod_df.set_index(join_on)["new_oai_mod"].apply(lambda d: d[] )[some_mod.values]
    .sort_index()
    .equals(merged_df.set_index(join_on)["new_sent_convo"].sort_index())
), "max scores off"
# %%
merged_df.set_index(join_on)["new_sent_convo"] == analysis_mod_df.set_index(join_on).iloc[
    : len(new_mod)
]["sent_convo"]
# d1 = analysis_mod_df.set_index(["mod_model", "conversation_id"])
# d2 = new_mod.set_index("conversation_id")
# d3 = default_mod.set_index("conversation_id")
# pd.concat([d2, d3], axis=1)
# %%
# join these on metada cols
out["mod_how_str"] = out["manipulation"].apply(lambda d: f"{ord(d['sep'])}_{d['kind']}")
print(out["new_max_score"].describe(), out["default_max_score"].describe())
return out
# %%
# del out["convo_str"]
# return out
# %%
# Basically nothing got moderated though?
analysis_mod_df.query('conversation_id=="74ef5861d9dc49de9dcf6484872d6bec"')["new_oai_mod"].apply(
    lambda d: max(d["results"][0]["category_scores"].values())
).value_counts()
# max 0.006, w and w/o manipulation
[max(d["category_scores"].values()) for d in chat_df["openai_moderation"].iloc[0]]
# is all 0.99s
# and content the same
assert [d["content"] for d in chat_df["conversation"].iloc[0]] == [
    d["content"]
    for d in analysis_mod_df.query('conversation_id=="74ef5861d9dc49de9dcf6484872d6bec"')[
        "sent_convo"
    ]
][-8:]

# %%
df = analysis_mod_df.head(10)
