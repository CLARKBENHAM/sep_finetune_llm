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
import math
from datetime import datetime
from pprint import pprint
from scipy.stats import binom_test

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
    if "results" in openai_moderation:
        openai_moderation = openai_moderation["results"]
    return max([max(m["category_scores"].values()) for m in openai_moderation])


def chat_max_by_cat(openai_moderation, categories=None):
    """Max score of any chat in convo by category"""
    if "results" in openai_moderation:
        openai_moderation = openai_moderation["results"]
    if categories is None:
        categories = list(openai_moderation[0]["category_scores"].keys())
    return {c: max((r["category_scores"][c] for r in openai_moderation)) for c in categories}


# can skip to l:225 "Start of Analysis"
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

MAX_MOD_ITEMS = 32
MAX_TURN_TOKENS = 4096  # not sure if real limitation
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
# WARN: makes requests


def _make_mod_request(mod_in, model):
    e2 = None
    for ix in range(5):
        try:
            if e2 is not None:
                time.sleep(15 + 4**ix)
            mod = client.moderations.create(input=mod_in, model=model)
        except Exception as e:
            e2 = e
        else:
            return mod.model_dump()  # ["results"]
    print(datetime.now(), e2)
    print(e2["response"])
    return None


def make_mod_requests(r):
    """
    Warn: scores will be slightly different
    if recover (insert single entries) vs. making it all yourself the first time
    but <0.05 pts off total
    """

    if (not r["new_oai_mod"]) or pd.isna(r["new_oai_mod"]):  # making full from empty row
        print(datetime.now(), "new row")
        return _make_mod_request([c["content"] for c in r["sent_convo"]], r["new_model"])
    n_results = max(
        len([m for m in r["new_oai_mod"] if m and pd.notna(m) and isinstance(m, list)]),
        len([m for m in r["new_oai_mod"]["results"] if m and pd.notna(m)]),
    )
    exp_results = len(r["sent_convo"])
    if n_results == exp_results:  # already complete row
        print("skipping")
        return r["new_oai_mod"]
    else:
        print("filling in parts, this will change data formating")
        assert False
        out = r["new_oai_mod"]
        if exp_results != n_results:
            print("WARN: tossing all previous")
            out = [None] * len(r["sent_convo"])
        out = [
            _make_mod_request(c["content"], r["new_model"]) if o is None else o
            for c, o in zip(r["sent_convo"], out)
        ]
        return out


def make_mod_requests_with_progress(args):
    index, total, r = args
    if index % (total // 25) == 0:
        print(f"Progress: {index / total * 100:.2f}% {datetime.now()}\n")
    return make_mod_requests(r)


check_mod_df = pd.read_pickle(f"data_dump/oai_mod/comp_results_{git_hash()}_full.pkl")
total = len(check_mod_df)
args = [(index, total, r) for index, r in enumerate(check_mod_df.to_dict("records"))]

with ThreadPoolExecutor(max_workers=2) as executor:
    check_mod_df["new_oai_mod"] = list(executor.map(make_mod_requests_with_progress, args))
    # o = list(executor.map(make_mod_requests_with_progress, args[:5]))

check_mod_df.to_pickle(f"data_dump/oai_mod/comp_results_{git_hash()}_full.pkl")
# %%
# Start of Analysis

analysis_mod_df = pd.read_pickle("data_dump/oai_mod/comp_results_ba0cefe.pkl")
print(f"Droping nans: {analysis_mod_df.isna().sum().sum()}")
analysis_mod_df = analysis_mod_df.dropna()

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
# %%
# skip if dropped nans
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
# %%
# de-dup convo_str's that were the same but from different conversations
print(analysis_mod_df.shape)

# cols should be unique on
uniq_cols = ["mod_model", "convo_str", "man_str"]

# What tried to set content mod model as
analysis_mod_df = analysis_mod_df.rename(columns={"new_model": "_sent_model"})

analysis_mod_df["mod_model"] = analysis_mod_df["new_oai_mod"].apply(lambda d: d["model"])
analysis_mod_df["convo_str"] = analysis_mod_df["sent_convo"].apply(str)
analysis_mod_df["man_str"] = analysis_mod_df["manipulation"].astype(str)
extra_dup_mask = analysis_mod_df.duplicated(subset=uniq_cols, keep="first")
if extra_dup_mask.sum() > 0:
    # Avg L1-norm amoung maxscore for duplicated values
    assert (
        analysis_mod_df[analysis_mod_df.duplicated(subset=uniq_cols, keep=False)]
        .groupby(uniq_cols)["new_oai_mod"]
        .transform(lambda x: x.apply(chat_max_scores))
        .apply(lambda x: np.sum(np.abs(x - np.mean(x))))
        .mean()
    ) == 0
    extra_dups = analysis_mod_df[extra_dup_mask]
    analysis_mod_df = analysis_mod_df[~extra_dup_mask]
    print(f"WARN: dropped {extra_dup_mask.sum()} duplicates based on {uniq_cols}")
    print(extra_dups.groupby(["mod_model", "man_str"]).size())
    print("Max Scores where dropped", extra_dups["new_oai_mod"].apply(chat_max_scores).describe())
# check dropped dups all
check_cols = ["mod_model", "convo_ix", "conversation_id", "man_str"]
assert analysis_mod_df[check_cols].drop_duplicates().shape[0] == analysis_mod_df.shape[0]
d = copy.deepcopy(analysis_mod_df)


# %%
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

# Shouldn't have to do this
print(f"WARN: dropping NAs: {merged_df.isna().any(axis=1).sum()}")
merged_df = merged_df.dropna()
merged_df["mod_how_str"] = merged_df["manipulation"].apply(lambda d: f"{ord(d['sep'])}_{d['kind']}")
merged_df["new_minus_default_max_score"] = (
    merged_df["new_max_score"] - merged_df["default_max_score"]
)
# %% Print summaries
print(merged_df["new_max_score"].describe(), merged_df["default_max_score"].describe())
print(
    merged_df.groupby("mod_how_str")["new_minus_default_max_score"]
    .agg(["mean", "sem"])
    .sort_values(by="mean")
)
categories = [
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "self-harm",
    "self-harm/instructions",
    "self-harm/intent",
    "sexual",
    "sexual/minors",
    "violence",
    "violence/graphic",
]
df = pd.concat([
    pd.DataFrame({
        c: (merged_df[f"new_{c}"] - merged_df[f"default_{c}"]).agg(
            ["mean", "sem", lambda d: np.mean(d) / d.sem()]
        )
    }).transpose()
    for c in categories
]).rename(columns={"<lambda>": "t-score"})
print(df.sort_values(by="t-score"))

print(
    "Percent of turns with max mod 10ppt lower from sep:"
    f" {100*np.mean(merged_df['new_minus_default_max_score'] <= -0.1):.1f}%"
)
print(
    "Percent of turns that got flagged that wouldn't have:"
    f" {100*np.mean(merged_df['new_any_flagged'] - merged_df['default_any_flagged'] > 0):.1f}%"
)
missed_flag_ix = merged_df["new_any_flagged"] - merged_df["default_any_flagged"] < 0
print(f"Percent of turns that should've got flagged but didn't: {100*np.mean(missed_flag_ix):.1f}%")
# %%
# Read the strings where adding seperators worked
not_flagged_convos = merged_df[missed_flag_ix]["default_sent_convo"]
print(not_flagged_convos.value_counts().value_counts().sort_index())
# 133 strs only work for 1, then ~30-40 work for 2-7

value_counts = not_flagged_convos.value_counts()
df = value_counts[value_counts > len(ORD_USE_BETWEEN) / 2].reset_index()
df.columns = ["index", "value_counts"]
df = df[["value_counts", "index"]]
df.to_csv("data_dump/oai_mod/mostly_passed_flagging.csv", sep="\t", index=False)

# %%
# compare the langauge where adding serperators mostly worked
chat_df = pd.read_pickle(f"data_dump/oai_mod/comparison_base_df18bd574.pkl")
cid2lang = chat_df[["conversation_id", "language"]].set_index("conversation_id")[
    "language"
]  # .to_dict("index")
merged_df["language"] = merged_df["conversation_id"].apply(lambda c: cid2lang[c])

lang_default = merged_df["language"].value_counts()
lang_missed_flag = merged_df["language"][missed_flag_ix].value_counts()
exp_lang_missed_flag = merged_df["language"].value_counts(normalize=True) * missed_flag_ix.sum()

results = {}
p = missed_flag_ix.sum() / len(merged_df)  # Success probability under null hypothesis
for language in lang_default.index:
    n = lang_default.loc[language]  # Number of trials
    k = lang_missed_flag.loc[language]  # Number of successes
    # Binomial test
    p_value = binom_test(k, n, p)
    results[language] = {"p_value": p_value, "ratio_change": (k / n) / p}

sig_langs = {lan: d for lan, d in results.items() if d["p_value"] < 0.001 / len(lang_default.index)}
sig_langs_df = pd.DataFrame(sig_langs.values())
sig_langs_df.index = sig_langs.keys()
sig_langs_df["num_missed_flagged"] = lang_missed_flag[sig_langs_df.index]
sig_langs_df["exp_num_missed_flagged"] = (
    exp_lang_missed_flag[sig_langs_df.index].round(0).astype(int)
)
with pd.option_context("display.float_format", "{:,.2e}".format):
    print(sig_langs_df.sort_values("p_value"))
print(
    lang_default.loc[sig_langs.keys()] / lang_default.sum(),
    lang_missed_flag.loc[sig_langs.keys()] / lang_missed_flag.sum(),
)

# %% Can you combine the conditions?
check_langs = ["Portuguese", "French", "unknown", "Russian"]
is_lang = merged_df["language"].isin(check_langs)
sep192 = merged_df["manipulation"].apply(lambda d: d["sep"] == chr(192))
ix = is_lang & sep192
print(
    f"Percent of turns that got flagged that wouldn't have on ({' or '.join(check_langs)}) and by"
    " sep 192:"
    f" {100*np.mean(merged_df['new_any_flagged'][ix] - merged_df['default_any_flagged'][ix] < 0):.1f}%"
)


# %%
# code validation: skip if dropped nans
assert (
    merged_df["new_sent_convo"].apply(type).value_counts()
    == merged_df["default_sent_convo"].apply(type).value_counts()
).all()
for c in join_on:
    assert (
        new_mod[c].value_counts().sort_index() == merged_df[c].value_counts().sort_index()
    ).all(), f"{c} new_mod vs merged differs"
    assert (
        merged_df[c].value_counts().sort_index() + default_mod[c].value_counts().sort_index()
        == analysis_mod_df[c].value_counts().sort_index()
    ).all(), f"{c} merged + default vs analsysi_mod_df differs"
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

mod_a = (
    analysis_mod_df["new_oai_mod"][some_mod]
    .apply(lambda d: chat_max_scores(d["results"]))
    .describe()
)
mod_m = merged_df["new_max_score"].describe()
assert mod_a.round(3).equals(mod_m.round(3))

# since the array sizes are different, drop count std
nmod_a = (
    analysis_mod_df["new_oai_mod"][~some_mod]
    .apply(lambda d: chat_max_scores(d["results"]))
    .describe()
    .drop(["count", "std"])
)
nmod_m = merged_df["default_max_score"].describe().drop(["count", "std"])
assert nmod_a.round(3).equals(nmod_m.round(3))

# return merged_df

# %%
from scipy.stats import ks_2samp
from matplotlib import colors
import hashlib


def str_to_color(string):
    hash_object = hashlib.md5(string.encode())
    # Take parts of the hash for hue, saturation, and lightness
    f360 = lambda s: (int(hash_object.hexdigest()[s], 16) % 360) / 360
    # Ensures sat and v are at least 50%
    f100_min50 = lambda s: 0.5 + (int(hash_object.hexdigest()[s], 16) % 50) / 100
    hue = f360(slice(0, 3))
    sat = f100_min50(slice(3, 5))
    v = f100_min50(slice(5, 7))
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
    ax.hist(
        data1,
        color=str_to_color(col1),
        alpha=0.5,
        label=col1 + f" m: {data1.mean():.3f} sem: {data1.sem():.3f}",
    )
    ax.hist(
        data2,
        color=str_to_color(col2),
        alpha=0.5,
        label=col2 + f" m: {data2.mean():.3f} sem: {data2.sem():.3f}",
    )
    statistic, p_value = ks_2samp(data1.dropna(), data2.dropna(), alternative="two-sided")
    title = f"{col1} vs {col2}"
    title += f"\nKS Statistic: {statistic:.3f}, P-Value: {p_value:.3f}"
    color = "red" if p_value < sig_level else "black"
    ax.set_title(title, color=color)
    ax.legend()
    return ax


data1 = merged_df["new_max_score"]
data2 = merged_df["default_max_score"]
fig, ax = plt.subplots(figsize=(10, 6))
ax = _ks_hist_plot(data1, data2, col1="with seperators", col2="w/o seperators", ax=ax)
fig.suptitle(f"{', '.join(merged_df['mod_model'].unique())} Max Category Score per Message")
fig.subplots_adjust(top=0.86)
fig.savefig(f"plots/oai_mod/average_max_scores_yn_seperators_{git_hash()}.png")
# %%
import scipy.stats as stats


def reg_plot(
    x1,
    y1,
    x_name=None,
    y_name=None,
    title=None,
):
    x_name, y_name = get_name(x1, x_name, "X"), get_name(y1, y_name, "Y")
    if title is None:
        title = f"{y_name} vs {x_name}"

    ax = sns.regplot(
        x=x1, y=y1, scatter=True, ci=95, line_kws={"color": "red"}, scatter_kws={"s": 2}
    )
    ax.set_title(title)
    ax.set_ylabel(y_name)
    ax.set_xlabel(x_name)
    corr, p = stats.pearsonr(x1, y1)
    ax.text(
        0.05,
        0.95,
        f"corr: {corr:.2f} p: {p:.2f}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    plt.tight_layout()
    plt.show()


def avg_by_bucket(X, Y, x_name=None, y_name=None, ax=None, by_width=False):
    """Bin X. Then plot average Y's for each bin of X"""
    x_name, y_name = get_name(X, x_name, "X"), get_name(Y, y_name, "Y")
    if ax is None:
        fig, ax = plt.subplots()
    # equal width buckets
    if by_width:
        buckets = pd.cut(X, bins=min(20, math.ceil((len(X) + 1) / 10)))
    else:
        # equal num element buckets
        buckets = pd.qcut(X, q=min(10, math.ceil((len(X) + 1) / 10)), duplicates="drop")
    bucket_means = pd.DataFrame({x_name: X, y_name: Y}).groupby(buckets)[y_name].mean()
    ax.bar(range(len(bucket_means)), bucket_means, color=str_to_color(y_name))
    ax.set_xticks(
        range(len(bucket_means)),
        [f"{interval.mid:.0f}" for interval in bucket_means.index],
        rotation=90,
    )
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.figure.tight_layout()
    return ax


def prompt_lengths_vs_max_score(df, by_width=False):
    """
    plot both prompt length and output length vs max mod score
    """
    prompt_lens = df["new_sent_convo"].apply(lambda d: num_tokens_from_messages([d]))
    prompt_lens.name = "Sent Convo Num Tokens"
    og_prompt_lens = df["default_sent_convo"].apply(lambda d: num_tokens_from_messages([d]))
    og_prompt_lens.name = "Original Convo Num Tokens"
    new_max_scores = copy.deepcopy(df["new_max_score"])
    new_max_scores.name = "Avg Max Mod with seperators"
    # reg_plot(og_prompt_lens, prompt_lens, "original len", "manipulation lens")
    reg_plot(og_prompt_lens, new_max_scores)
    reg_plot(prompt_lens, new_max_scores)

    score_diff = new_max_scores - df["default_max_score"]
    score_diff.name = "Avg Max Mod with seperators - w/o seperators"
    reg_plot(og_prompt_lens, score_diff)

    # Average mod by prompt len
    ax = avg_by_bucket(prompt_lens, score_diff, by_width=by_width)
    plt.show()

    ax = avg_by_bucket(og_prompt_lens, score_diff, by_width=by_width)
    plt.show()


prompt_lengths_vs_max_score(merged_df)
# %%
# Longest prompts have bigger difference
i = 800
gt_800_tokens = merged_df["default_sent_convo"].apply(lambda d: num_tokens_from_messages([d])) > i
print(i)
fig, ax = plt.subplots(figsize=(7, 5))
_ks_hist_plot(data1[gt_800_tokens], data2[gt_800_tokens], ax=ax)
fig.suptitle(f"Original convo turn was >{i} tokens")
fig.subplots_adjust(top=0.86)
fig.savefig(f"plots/oai_mod/average_max_scores_yn_seperators_{git_hash()}_gt_800.png")
# %%


def plot_comparisons(df, cat_col, scores, comparison_type="categorical", sig_level=0.01):
    """
    Generate comparisons for different categories or scores as lower triangle

    :param df: Pandas DataFrame with the data.
    :param columns: List of columns for comparisons.
    :param score_column: Column name of the numeric scores to compare.
    :param comparison_type: Type of comparison - 'categorical' or 'score'.
    """
    if isinstance(scores, str):
        scores = df[scores]
    categories = df[cat_col].unique()
    n = len(categories)
    print(n, categories)
    fig, axs = plt.subplots(n, n, figsize=(5 + 3 * n, 5 + 3 * n))
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            ax = axs[i, j]
            if j < i:
                # Comparing numeric scores for two different categories
                if comparison_type == "categorical":
                    # Comparing scores within categories
                    data1 = scores[df[cat_col] == cat1]
                    data2 = scores[df[cat_col] == cat2]
                else:
                    assert False
                    # Comparing scores across different columns
                    data1 = scores
                    data2 = scores
                _ks_hist_plot(data1, data2, col1=cat1, col2=cat2, ax=ax, sig_level=sig_level)
            else:
                ax.set_visible(False)

    fig.tight_layout()
    # label rows
    for y, cat in enumerate(categories):
        pos = axs[y, 0].get_position()
        x0, y0, x1, y1 = [getattr(pos, i) for i in "x0, y0, x1, y1".split(", ")]
        fig.text(-0.01, (y0 + y1) / 2, cat, va="center", fontsize=12, rotation="vertical")
    # label cols
    for x, cat in enumerate(categories):
        pos = axs[0, x].get_position()
        x0, y0, x1, y1 = [getattr(pos, i) for i in "x0, y0, x1, y1".split(", ")]
        fig.text(
            (x0 + x1) / 2,
            -0.01,
            cat,
            ha="center",
            fontsize=12,
        )
    fig.tight_layout()
    fig.show()
    return fig


# # No difference in which seperator tokens which work best/worst for which categories
for c in list(merged_df["new_oai_mod"].iloc[0]["results"][0]["category_scores"].keys())[1:]:
    diff = merged_df[f"new_{c}"] - merged_df[f"default_{c}"]
    fig = plot_comparisons(merged_df, "mod_how_str", diff)
    fig.suptitle(
        f"Compare different preprocessing steps on difference in {c} numeric scores",
        fontsize=50,
    )
    path = f"plots/oai_mod/compare_sep_tokens_on_oai_mod_{git_hash()}"
    if not os.path.exists(path):
        os.mkdir(path)
    fig.savefig(
        f"{path}/{c.replace('/', '')}.png",
        facecolor="w",
        bbox_inches="tight",
    )

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
