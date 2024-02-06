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
    final_chat_df,
    ord_vals=ORD_USE_BETWEEN + [None],
    model="text-moderation-latest",
    make_new_convo=None,
):
    # only valid model https://platform.openai.com/docs/models/moderation as of Feb 05
    # is "text-moderation-007" but errors if don't use a '*-stable' or '*-latest'
    # if model not in ("text-moderation-007"):
    if model not in ("text-moderation-007", "text-moderation-stable", "text-moderation-latest"):
        print(f"WARN: model {model} not expected")

    new_dfs = []
    for ord_val in ord_vals:
        _r_df = pd.DataFrame(index=final_chat_df.index)
        _r_df["conversation_id"] = final_chat_df["conversation_id"]  # should've added
        _r_df["new_completion"] = None
        _r_df["new_oai_mod"] = None
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
                assert False
                _r_df["sent_convo"] = _r_df.apply(make_new_convo, axis=1)
        new_dfs += [_r_df]
    return pd.concat(new_dfs)


chat_df = pd.read_pickle(f"data_dump/oai_mod/comparison_base_df18bd574.pkl")
check_mod_df = pd.concat([
    # make_results_frame(chat_df, model="text-moderation-007"),
    make_results_frame(chat_df, model="text-moderation-latest"),  # works with current package
])
del check_mod_df["new_completion"]
# %%
MAX_MOD_ITEMS = 32
MAX_TURN_TOKENS = 4096  # not sure if real
MAX_MOD_TOKENS = 32768 - 5
check_mod_df["sent_convo"] = check_mod_df["sent_convo"].apply(
    lambda l: end_of_convo(
        [
            {
                **c,
                "content": (
                    take_last_tokens(c["content"], MAX_TURN_TOKENS)
                    if len(c["content"]) > MAX_TURN_TOKENS * 1.5
                    else c["content"]
                ),
            }
            for c in l[-MAX_MOD_ITEMS:]
        ],
        max_tokens=MAX_MOD_TOKENS,
    )
)
check_mod_df.to_pickle(f"data_dump/oai_mod/comparison_base_check_mod_df{git_hash()}.pkl")
print(check_mod_df["sent_convo"].apply(num_tokens_from_messages).describe())
# %%
import math
from datetime import datetime


def _make_mod_request(mod_in, model):
    e2 = None
    for ix in range(4):
        try:
            mod = client.moderations.create(input=mod_in, model=model)
        except Exception as e:
            time.sleep(15 + 3**ix)
            e2 = e
        else:
            return mod.model_dump()  # ["results"]
    print(e2)
    return None


def make_mod_requests(r):
    """
    Warn: scores will be slightly different
    if recover (insert single entries) vs. making it all yourself the first time
    but <0.05 pts off total
    """
    if (not r["new_oai_mod"]) or (
        isinstance(r["new_oai_mod"], float) and math.isnan(r["new_oai_mod"])
    ):
        return _make_mod_request([c["content"] for c in r["sent_convo"]], r["new_model"])
    out = r["new_oai_mod"]
    if len(out) != len(r["sent_convo"]):
        print("tossing")
        out = [None] * len(r["sent_convo"])
    out = [
        _make_mod_request(c["content"], r["new_model"]) if o is None else o
        for c, o in zip(r["sent_convo"], out)
    ]
    return out


def make_mod_requests_with_progress(args):
    index, total, r = args
    if index % (total // 25) == 0:  # Update every 4%
        print(f"Progress: {index / total * 100:.2f}% {datetime.now()}")
    return make_mod_requests(r)


total = len(check_mod_df)
args = [(index, total, r) for index, r in enumerate(check_mod_df.to_dict("records"))]

with ThreadPoolExecutor(max_workers=3) as executor:
    check_mod_df["new_oai_mod"] = list(executor.map(make_mod_requests_with_progress, args))

check_mod_df.to_pickle(f"data_dump/oai_mod/comp_results_{git_hash()}.pkl")
# %%
analysis_mod_df = pd.read_pickle("data_dump/oai_mod/comp_results_0775b2e.pkl")
analysis_mod_df["new_oai_mod"] = analysis_mod_df["new_oai_mod"].apply(
    lambda d: [{**d, "results": [r]} for r in d["results"]]
)
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
num_made = (
    analysis_mod_df["new_model"].nunique() * analysis_mod_df["manipulation"].apply(str).nunique()
)
assert d.sent_convo.apply(len).sum() == len(analysis_mod_df)
# #only for this subset
# if 'chat_df' in locals():
#    assert chat_df['conversation'].apply(len).sum() * num_made == len(analysis_mod_df)
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
assert analysis_mod_df.groupby(["conversation_id", "convo_ix"]).ngroups * num_made == len(
    analysis_mod_df
)
assert (
    d.sent_convo.apply(len).value_counts().sort_index() / num_made
    == (analysis_mod_df.groupby("conversation_id")["convo_ix"].max() + 1)
    .value_counts()
    .sort_index()
).all()

# del d
print(analysis_mod_df.shape)

# cols should be unique on: maybe dup strings; def dup models
uniq_cols = ["mod_model", "convo_str", "man_str"]

# What tried to set content mod model as
analysis_mod_df = analysis_mod_df.rename(columns={"new_model": "_sent_model"})

# analysis_mod_df = analysis_mod_df.query("_sent_model=='text-moderation-latest'")

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
keep_cols = join_on + ["sent_convo", "new_oai_mod"]
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
).rename(columns={"sent_convo": "default_sent_convo", "new_oai_mod": "default_oai_mod"})

merged_df = (
    new_mod.set_index(join_on)
    .merge(default_mod.set_index(join_on), left_index=True, right_index=True, how="left")
    .reset_index()
)
print(merged_df.shape)
# %%
assert (
    merged_df["new_sent_convo"].apply(type).value_counts()
    == merged_df["default_sent_convo"].apply(type).value_counts()
).all()
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
assert len(set(merged_df["new_sent_convo"].apply(lambda d: d["content"]))) == len(merged_df)
# added below since counts were wrong above if used .nunique() not len(set(
assert (
    analysis_mod_df.set_index(join_on)["sent_convo"][~some_mod.values]
    .apply(lambda d: str(d["content"]))
    .sort_index()
    .values
    == (
        merged_df.set_index(join_on)["default_sent_convo"]
        .sort_index()
        .apply(lambda d: str(d["content"]))
        .unique()
    )
).all(), "default_sent_convo off"

assert (
    analysis_mod_df.set_index(join_on)["new_oai_mod"]
    .apply(lambda d: d["id"])[some_mod.values]
    .sort_index()
    .equals(merged_df.set_index(join_on)["new_oai_mod"].apply(lambda d: d["id"]).sort_index())
), "new_oai_mod id's off"
assert (
    analysis_mod_df.set_index(join_on)["new_oai_mod"]
    .apply(lambda d: chat_max_scores(d["results"]))[some_mod.values]
    .sort_index()
    .equals(
        merged_df.set_index(join_on)["new_oai_mod"]
        .apply(lambda d: chat_max_scores(d["results"]))
        .sort_index()
    )
), "oai_mod max scores off"

merged_df["mod_how_str"] = merged_df["manipulation"].apply(lambda d: f"{ord(d['sep'])}_{d['kind']}")
# return merged_df
# %%
mod_a = (
    analysis_mod_df["new_oai_mod"][some_mod]
    .apply(lambda d: chat_max_scores(d["results"]))
    .describe()
)
mod_m = merged_df["new_max_score"].describe()
assert mod_a.round(3).equals(mod_m.round(3))

# drop std since the array sizes are different
nmod_a = (
    analysis_mod_df["new_oai_mod"][~some_mod]
    .apply(lambda d: chat_max_scores(d["results"]))
    .describe()
    .drop(["count", "std"])
)
nmod_m = merged_df["default_max_score"].describe().drop(["count", "std"])
assert nmod_a.round(3).equals(nmod_m.round(3))
# %%
print(merged_df["new_max_score"].describe(), merged_df["default_max_score"].describe())
print(merged_df.groupby("mod_how_str")["new_max_score"].describe())

# %%
from scipy.stats import ks_2samp
from matplotlib import colors
import hashlib


def str_to_color(string):
    hash_object = hashlib.md5(string.encode())
    # Take parts of the hash for hue, saturation, and lightness
    # hue = int(hash_object.hexdigest()[:3], 16) % 360  # Hue: 0-360
    # sat = int(hash_object.hexdigest()[3:5], 16) % 101  # Saturation: 0-100%
    # light = int(hash_object.hexdigest()[5:7], 16) % 101  # Lightness: 0-100%
    # return f"hsl({hue}, {sat}%, {light}%)"

    f = lambda s: (int(hash_object.hexdigest()[s], 16) % 100) / 100
    hue = f(slice(0, 2))
    f_min50 = (
        lambda s: 0.5 + (int(hash_object.hexdigest()[s], 16) % 50) / 100
    )  # Ensures sat and v are at least 50%
    sat = f_min50(slice(2, 4))
    v = f_min50(slice(4, 6))
    return colors.hsv_to_rgb((hue, sat, v))


def get_name(d, n, default=""):
    if n is None:
        n = getattr(d, "name", None)
    if n is None:
        n = getattr(d, "columns", [None])[0]
    if n is None:
        n = default
    return n


def _ks_hist_plot(data1, data2, col1=None, col2=None, ax=None, sig_level=0.05):
    if ax is None:
        fig, ax = plt.subplots()

    col1, col2 = get_name(data1, col1, "1"), get_name(data2, col2, "2")
    # sns.histplot(data1, color=str_to_color(col1), alpha=0.5, label=col1, ax=ax)
    # sns.histplot(data2, color=str_to_color(col2), alpha=0.5, label=col2, ax=ax)
    ax.hist(
        data1,
        color=str_to_color(col1),
        alpha=0.5,
        label=col1 + f" m: {data1.mean():.2f} sem: {data1.sem():.2f}",
        # density=True,
    )
    ax.hist(
        data2,
        color=str_to_color(col2),
        alpha=0.5,
        label=col2 + f" m: {data2.mean():.2f} sem: {data2.sem():.2f}",
        # density=True,
    )
    statistic, p_value = ks_2samp(data1.dropna(), data2.dropna(), alternative="two-sided")
    title = f"{col1} vs {col2}"
    title += f"\nKS Statistic: {statistic:.3f}, P-Value: {p_value:.3f}"
    color = "red" if p_value < sig_level else "black"
    ax.set_title(title, color=color)
    ax.legend()
    return ax


_ks_hist_plot(data1=merged_df["new_max_score"], data2=merged_df["default_max_score"])
# print(data1.agg(["mean", "sem"]), data2.agg(["mean", "sem"]))
# %%
# %%
p = np.arange(0, 1, 0.05)
print(
    analysis_mod_df["new_oai_mod"][some_mod]
    .apply(lambda d: chat_max_scores(d["results"]))
    .describe(percentiles=p),
    analysis_mod_df["new_oai_mod"][~some_mod]
    .apply(lambda d: chat_max_scores(d["results"]))
    .describe(percentiles=p),
)

# %%
df = analysis_mod_df.head(10)

# %%
# assert merged_df["new_sent_convo"].apply(lambda d: d["content"]).nunique() == len(merged_df)
# Don't know why .nunique != len(set())
s, e = 0, len(merged_df)
m = (s + e) // 2
while s != m and m != e:
    df = merged_df["new_sent_convo"].iloc[s:m]
    if df.apply(lambda d: d["content"]).nunique() < len(df):
        e = m - 1
        print("good", s, m)
    else:
        s = m + 1
    m = (s + e) // 2
    print(s, m, e)
s, m = 0, 48
df = merged_df["new_sent_convo"][s:m]
print(
    df.apply(lambda d: d["content"]).nunique(), len(set(df.apply(lambda d: d["content"]))), len(df)
)
df.to_pickle("data_dump/data_df_where_content_nunique_doesnt_match_set_of_content")

# %%
# Check equal if it matters to send the whole conversation in at once or in pieces
import random

for d in random.sample(check_mod_df.to_dict("records"), 10):
    r2 = client.moderations.create(input=[i["content"] for i in d["sent_convo"]])
    r2s = [client.moderations.create(input=i["content"]) for i in d["sent_convo"]]
    score1 = np.array([max(res.category_scores.model_dump().values()) for res in r2.results])
    score2 = np.array(
        [max(res["category_scores"].values()) for _r2 in r2s for res in _r2.model_dump()["results"]]
    )
    print(np.abs(score1 - score2).sum())
    # assertion triggers
    assert (np.around(score1, 2) == np.around(score2, 2)).all(), d["sent_convo"]
# Scores are mostly the same, but not identical

# d=check_mod_df.iloc[-1:].to_dict("records")[0]
# r=client.moderations.create(input=d['sent_convo'][0]['content'])
# print(max(r.results[0].category_scores.model_dump().values()))

# %% Random Scrape
duplicate_mask = d.duplicated(subset=uniq_cols, keep="first")
dropped_rows = analysis_mod_df[duplicate_mask]
print(dropped_rows)
# This makes dups, shouldn't have any dups
# Change this to filter all dups at the start
analysis_mod_df.query("conversation_id=='82af34c6bdee4cf18d5e3b203ce5fefa' and convo_ix==9")[
    "sent_convo"
]
# %% Scrape
r2 = client.moderations.create(input=[i["content"] for i in d["sent_convo"]])
print(len(r2.results))
print(max(r2.results[0].category_scores.model_dump().values()))
print([max(res.category_scores.model_dump().values()) for res in r2.results])
# %%
# r2s = [client.moderations.create(input=i['content']) for i in d['sent_convo']]
# print(max(r2s.results[0].category_scores.model_dump().values()))
print([max(res["category_scores"].values()) for _r2 in r2s for res in _r2.model_dump()["results"]])
# %%
o = make_mod_requests(d)
print(o)
print([len(i["results"]) for i in o])
print([max(i["results"][0]["category_scores"].values()) for i in o])
# %%
