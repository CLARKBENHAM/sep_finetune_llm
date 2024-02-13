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
from scipy.stats import binomtest
from scipy.spatial.distance import cosine
import openai
import backoff

# from pyarrow import parquet as pq
from concurrent.futures import ThreadPoolExecutor
import os
import copy
from openai import OpenAI
import gc
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
    take_first_tokens,
)

api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

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


def chat_is_flagged(openai_moderation):
    """Expects list. If any message in convo is flagged"""
    if "results" in openai_moderation:
        openai_moderation = openai_moderation["results"]
    return any((r["flagged"] for r in openai_moderation))


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


# can skip to l:525 "merge_df" for analysis section, will need make_
# %%
# can skip
files = [
    "data_dump/lmsys-chat-1m/train-00000-of-00006-4feeb3f83346a0e9.parquet",
    "data_dump/lmsys-chat-1m/train-00001-of-00006-4030672591c2f478.parquet",
    "data_dump/lmsys-chat-1m/train-00002-of-00006-1779b7cec9462180.parquet",
    "data_dump/lmsys-chat-1m/train-00003-of-00006-2fa862bfed56af1f.parquet",
    "data_dump/lmsys-chat-1m/train-00004-of-00006-18f4bdd50c103e71.parquet",
    "data_dump/lmsys-chat-1m/train-00005-of-00006-fe1acc5d10a9f0e2.parquet",
]
all_chat_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

# sort chat_df by chat_max_scores on 'openai_moderation' column
all_chat_df = all_chat_df.sort_values(
    by="openai_moderation", key=lambda s: s.apply(chat_max_scores), ascending=False
).reset_index(drop=True)

chat_df = all_chat_df.head(1000)

check_langs = ["Portuguese", "French", "unknown", "Russian"]
chat_df2 = all_chat_df.iloc[1000:]
chat_df2 = chat_df2[chat_df2["language"].isin(check_langs)]
# Get the top 100 entries for each language
chat_df2 = chat_df2.groupby("language").apply(lambda x: x.head(100)).reset_index(drop=True)

# from not already used; will run 500 convos then compare at mod scores top 10% of turns by embedding cos dist
# only running through all 500 convos to keep code the same
chat_df3 = all_chat_df.iloc[1000:]
chat_df3 = chat_df3[~chat_df3["conversation_id"].isin(chat_df2["conversation_id"])].iloc[:500]


def parallel_apply(df, func, n_jobs):
    """
    Apply a function in parallel to the DataFrame.
    n_jobs: use len(os.sched_getaffinity(0)) on unix. os.cpu_count() is hyperthreads not physical
    """
    # Split the dataframe into even chunks to be processed in parallel
    df_split = np.array_split(df, n_jobs)

    # Use joblib to run the function in parallel
    df = pd.concat(
        Parallel(n_jobs=n_jobs)(
            delayed(lambda subset: subset.apply(func))(chunk) for chunk in df_split
        )
    )
    return df


# encoding._special_tokens
st = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}


def not_special(conversation, special=list(st.keys())):
    return not any((k in message["content"] for message in conversation for k in special))


# To see if can find a great number of tokens to smuggle into finetuning
used_convo_id = set(np.concatenate([d["conversation_id"].unique() for d in (chat_df3, chat_df2)]))
chat_df4 = all_chat_df.iloc[1000:]
chat_df4 = chat_df4[~chat_df4["conversation_id"].isin(used_convo_id)].iloc[:50000]

# had 4 special tokens out of 50k: '<|endoftext|>'
chat_df4 = chat_df4[parallel_apply(chat_df4["conversation"], not_special, n_jobs=8)]
chat_df4.to_pickle(f"data_dump/oai_mod/big_bad_finetune_chat_df4_{git_hash()}.pkl")
# parallel_apply(chat_df4["conversation"].head(1000), num_tokens_from_messages, n_jobs=8).describe()  # Errors (?)
print(chat_df4["conversation"].apply(num_tokens_from_messages).describe())
# 5.05M tokens right now


del all_chat_df
gc.collect()
# %%
# chat_df.to_pickle(f"data_dump/oai_mod/comparison_base_df{git_hash()}.pkl")
# chat_df2.to_pickle(f"data_dump/oai_mod/comparison_base_df{git_hash()}_lang_checks.pkl")
# chat_df3.to_pickle(f"data_dump/oai_mod/_temp_comparison_base_df{git_hash()}_base3.pkl")
# chat_df4.to_pickle(f"data_dump/oai_mod/_temp_comparison_base_df{git_hash()}_base3.pkl")

chat_df = pd.read_pickle("data_dump/oai_mod/comparison_base_check_mod_df0775b2e.pkl")
chat_df2 = pd.read_pickle("data_dump/oai_mod/comparison_base_df53ca02c_lang_checks.pkl")
chat_df3 = pd.read_pickle("data_dump/oai_mod/_temp_comparison_base_df8a70b20_base3.pkl")
chat_df4 = pd.read_pickle("data_dump/oai_mod/big_bad_finetune_chat_df4_2ae7e91.pkl")
# %% Define functions


def make_results_frame(
    final_chat_df,
    ord_vals=ORD_USE_BETWEEN + [None],
    model="text-moderation-latest",
    make_new_convo=None,
    keep_old_oai_mod=False,
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
        _r_df["new_model"] = model
        _r_df["new_oai_mod"] = None
        if keep_old_oai_mod:
            _r_df["og_openai_moderation"] = final_chat_df["openai_moderation"]

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


def cut(
    check_mod_df,
    col="sent_convo",
    MAX_MOD_ITEMS=32,
    MAX_TURN_TOKENS=4096,  # not sure if real limitation on moderation
    MAX_MOD_TOKENS=32768 - 5,  # model w shortest context window
    strip_first_assistant=True,
):
    check_mod_df[col] = check_mod_df[col].apply(
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
            strip_first_assistant=strip_first_assistant,
        )
    )
    return check_mod_df


def _make_mod_request(mod_in, model):
    e2 = None
    for ix in range(5):
        try:
            if e2 is not None:
                time.sleep(5 + 3**ix)
            mod = client.moderations.create(input=mod_in, model=model)
        except Exception as e:
            e2 = e
        else:
            return mod.model_dump()  # ["results"]
    print(datetime.now(), e2)
    return None


def make_mod_requests(r):
    """
    Warn: scores will be slightly different
    if recover (insert single entries) vs. making it all yourself the first time
    but <0.05 pts off total
    """

    if (not r["new_oai_mod"]) or pd.isna(r["new_oai_mod"]):  # making full from empty row
        # print(datetime.now(), "new row")
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
            _make_mod_request(result_col["content"], r["new_model"]) if o is None else o
            for result_col, o in zip(r["sent_convo"], out)
        ]
        return out


@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=8)
def get_embeddings(r, embeding_model="text-embedding-3-large"):
    if not isinstance(r[0], str):
        assert len(r) == 1, r
        r = r[0]["content"]
    return client.embeddings.create(model=embeding_model, input=r, encoding_format="float")


def make_async_reqs(df, max_workers=4, fn=make_mod_requests):
    total = len(df)
    args = [(index, total, r) for index, r in enumerate(df.to_dict("records"))]

    def req_progress(args):
        index, total, r = args
        out = fn(r)
        if total > 25 and index % (total // 25) == 0:
            print(f"Progress: {index / total * 100:.1f}% {datetime.now()}\n")
        return out

    print(f"Starting: {datetime.now()}\n")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        out = list(executor.map(req_progress, args))
    return out


# %%
chat_df = pd.read_pickle(f"data_dump/oai_mod/comparison_base_df18bd574.pkl")
check_mod_df = pd.concat([
    # make_results_frame(chat_df, model="text-moderation-007"),
    make_results_frame(chat_df, model="text-moderation-latest"),
])  # only *-latest model works with current package
check_mod_df2 = pd.concat([
    make_results_frame(chat_df2, ord_vals=[192, 8, None], model="text-moderation-latest"),
])

check_mod_df = cut(check_mod_df)
check_mod_df2 = cut(check_mod_df2)
check_mod_df.to_pickle(f"data_dump/oai_mod/comparison_base_check_mod_df{git_hash()}.pkl")
print(check_mod_df["sent_convo"].apply(num_tokens_from_messages).describe())
check_mod_df2.to_pickle(
    f"data_dump/oai_mod/comparison_base_check_mod_df{git_hash()}_lang_checks.pkl"
)
print(check_mod_df2["sent_convo"].apply(num_tokens_from_messages).describe())

# %%
# WARN: makes requests
check_mod_df = pd.read_pickle(f"data_dump/oai_mod/comp_results_{git_hash()}_full.pkl")
check_mod_df["new_oai_mod"] = make_async_reqs(check_mod_df, max_workers=4)
check_mod_df.to_pickle(f"data_dump/oai_mod/comp_results_{git_hash()}_full.pkl")

check_mod_df2["new_oai_mod"] = make_async_reqs(check_mod_df2, max_workers=5)
check_mod_df2.to_pickle(f"data_dump/oai_mod/comp_results_{git_hash()}_lang_check.pkl")
check_mod_df2["new_oai_mod"] = make_async_reqs(check_mod_df2, max_workers=2)
check_mod_df2.to_pickle(f"data_dump/oai_mod/comp_results_{git_hash()}_lang_check.pkl")
# didn't get anything new
# check_mod_df2["new_oai_mod"] = make_async_reqs(check_mod_df2, max_workers=1)
# check_mod_df2.to_pickle(f"data_dump/oai_mod/comp_results_{git_hash()}_lang_check.pkl")


# %%
# Analysis munging functions
def _split_convo_into_sep_turn_rows(
    analysis_mod_df,
    cols2split=["sent_convo", "new_oai_mod"],
    cols_as_list=[
        "sent_convo",
    ],
    run_asserts=True,
):
    """Given a long conversation and results arrays new_oai_mod
    split so each individual saying is 1 row
    adds "convo_ix" column to track ix in original conversation of id
    """
    # explode each turn in 'new_oai_mod' and 'sent_convo' into their own rows
    if run_asserts:
        d = copy.deepcopy(analysis_mod_df)
    analysis_mod_df = analysis_mod_df.copy()
    cols_as_list_ix = set([ix for ix, c in enumerate(cols2split) if c in cols_as_list])
    analysis_mod_df["paired"] = analysis_mod_df.apply(
        lambda row: [
            (ix, *[[v] if ix in cols_as_list_ix else v for ix, v in enumerate(tup)])
            for ix, tup in enumerate(zip(*[row[c] for c in cols2split]))
        ],
        axis=1,
    )
    analysis_mod_df = analysis_mod_df.explode("paired")
    analysis_mod_df[["convo_ix", *cols2split]] = pd.DataFrame(
        analysis_mod_df["paired"].tolist(), index=analysis_mod_df.index
    )
    analysis_mod_df = analysis_mod_df.drop(columns=["paired"])
    if run_asserts:
        # skip if dropped nansk
        # check exploded correctly
        num_made = (
            analysis_mod_df["new_model"].nunique()
            * analysis_mod_df["manipulation"].apply(str).nunique()
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
    return analysis_mod_df


def _make_dup_free(analysis_mod_df):
    """
    de-dup convo_str's that were the same but from different conversations
    also de-dup on models since what sent as isn't what get results as
    """
    # cols should be unique on
    uniq_cols = ["mod_model", "convo_str", "man_str"]
    extra_dup_mask = analysis_mod_df.duplicated(subset=uniq_cols, keep="first")
    if extra_dup_mask.sum() > 0:
        # Avg L1-norm amoung maxscore for duplicated values
        # assert (
        #    analysis_mod_df[analysis_mod_df.duplicated(subset=uniq_cols, keep=False)]
        #    .groupby(uniq_cols)["new_oai_mod"]
        #    .transform(lambda x: x.apply(chat_max_scores))
        #    .apply(lambda x: np.sum(np.abs(x - np.mean(x))))
        #    .mean()
        # ) == 0
        extra_dups = analysis_mod_df[extra_dup_mask]
        analysis_mod_df = analysis_mod_df[~extra_dup_mask].copy()
        print(f"WARN: dropped {extra_dup_mask.sum()} duplicates based on {uniq_cols}")
        print(extra_dups.groupby(["mod_model", "man_str"]).size())
        print(
            "Max Scores where dropped", extra_dups["new_oai_mod"].apply(chat_max_scores).describe()
        )
    # check dropped all dups
    check_cols = ["mod_model", "convo_ix", "conversation_id", "man_str"]
    assert analysis_mod_df[check_cols].drop_duplicates().shape[0] == analysis_mod_df.shape[0]
    return analysis_mod_df


# split into x and y values
def _explode_moderation_results(df, prefix, keep=None):
    """
    Explode moderation results into separate columns.
    :param df: DataFrame containing the moderation results.
    :param prefix: Prefix for the new columns.
    :return: DataFrame with exploded moderation results.
        drops new_completion and new_oai_mod columns
    """
    assert (df["new_oai_mod"].apply(lambda d: len(d["results"]) == 1)).all()
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


def munge_check_mod_df(result_df, chat_df):
    dropped_nans = result_df.isna().sum().sum()
    print(f"Droping nans: {dropped_nans}")
    result_df = result_df.dropna().copy()  # get a "setting slice on loc" if dont copy(?)

    result_df["new_oai_mod"] = result_df["new_oai_mod"].apply(
        lambda d: [{**d, "results": [r]} for r in d["results"]]
    )
    assert (result_df.sent_convo.apply(len) == result_df.new_oai_mod.apply(len)).all()

    result_df = _split_convo_into_sep_turn_rows(result_df, run_asserts=dropped_nans == 0)
    print(result_df.shape)

    # What tried to set content mod model as vs. what it actually is
    result_df = result_df.rename(columns={"new_model": "_sent_model"})
    result_df["mod_model"] = result_df["new_oai_mod"].apply(lambda d: d["model"])
    result_df["convo_str"] = result_df["sent_convo"].astype(str)
    result_df["man_str"] = result_df["manipulation"].astype(str)

    result_df = _make_dup_free(result_df)
    print("after dedupping", result_df.shape)

    # Move default row completions to being columns as Y
    join_on = ["conversation_id", "convo_ix", "mod_model"]
    keep_cols = join_on + ["sent_convo", "new_oai_mod"]
    new_only_cols = keep_cols + ["manipulation"]
    print(f"Dropping columns: {set(result_df.columns) - set(new_only_cols)}")
    some_mod = result_df["manipulation"].apply(
        lambda d: d["sep"] is not None or d["kind"] is not None
    )
    new_mod = _explode_moderation_results(result_df[some_mod], "new", keep=new_only_cols).rename(
        columns={"sent_convo": "new_sent_convo"}
    )
    default_mod = _explode_moderation_results(
        result_df[~some_mod], "default", keep=keep_cols
    ).rename(columns={"sent_convo": "default_sent_convo", "new_oai_mod": "default_oai_mod"})

    # already uniq on convo_str so conversation_id and convo_ix are unique here, aside from sep
    merged_df = (
        new_mod.set_index(join_on)
        .merge(default_mod.set_index(join_on), left_index=True, right_index=True, how="left")
        .reset_index()
    )
    if merged_df.isna().any(axis=1).sum() > 0:
        print(f"WARN: unexpected dropping NAs from merge_df: {merged_df.isna().any(axis=1).sum()}")
    merged_df = merged_df.dropna()
    print(merged_df.shape)

    # last cols to add
    merged_df["mod_how_str"] = merged_df["manipulation"].apply(
        lambda d: f"{ord(d['sep'])}_{d['kind']}"
    )
    merged_df["new_minus_default_max_score"] = (
        merged_df["new_max_score"] - merged_df["default_max_score"]
    )
    cid2lang = chat_df[["conversation_id", "language"]].set_index("conversation_id")["language"]
    merged_df["language"] = merged_df["conversation_id"].apply(lambda c: cid2lang[c])

    # Code validation. merged_df is done
    # TODO Should count be dropped here? Some analysis mod isn't be copied around correctly
    mod_a = (
        result_df["new_oai_mod"][some_mod]
        .apply(lambda d: chat_max_scores(d["results"]))
        .describe()
        .drop("count")
    )
    mod_m = merged_df["new_max_score"].describe().drop("count")
    assert mod_a.round(3).equals(mod_m.round(3)), mod_a.round(3).compare(mod_m.round(3))
    # since the array sizes are different, drop count std
    # these will be more off as dup's dropped from default_max_score
    # also change percentiles. Can hack atol
    nmod_a = (
        result_df["new_oai_mod"][~some_mod]
        .apply(lambda d: chat_max_scores(d["results"]))
        .describe()
        .drop(["count", "std"])
    )
    nmod_m = merged_df["default_max_score"].describe().drop(["count", "std"])
    assert np.allclose(nmod_a, nmod_m, atol=0.01 + 0.04 * (dropped_nans > 0)), nmod_a.round(
        2
    ).compare(nmod_m.round(2))

    # hack: skip if dropped nans. Should check what varies because nans
    if dropped_nans == 0:
        assert (
            merged_df["new_sent_convo"].apply(type).value_counts()
            == merged_df["default_sent_convo"].apply(type).value_counts()
        ).all()
        for c in join_on:
            assert (
                new_mod[c].value_counts().sort_index() == merged_df[c].value_counts().sort_index()
            ).all(), f"{c} new_mod vs merged differs"
            assert (
                merged_df[c].value_counts().sort_index()
                + default_mod[c].value_counts().sort_index()
                == result_df[c].value_counts().sort_index()
            ).all(), f"{c} merged + default vs analsysi_mod_df differs"
        assert (
            result_df.set_index(join_on)["sent_convo"][some_mod.values]
            .sort_index()
            .equals(merged_df.set_index(join_on)["new_sent_convo"].sort_index())
        ), "new convo off"
        assert len(set(merged_df["new_sent_convo"].apply(lambda d: d["content"]))) == len(merged_df)
        # added below since counts were wrong above if used .nunique() not len(set(
        assert (
            result_df.set_index(join_on)["sent_convo"][~some_mod.values]
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
            result_df.set_index(join_on)["new_oai_mod"]
            .apply(lambda d: d["id"])[some_mod.values]
            .sort_index()
            .equals(
                merged_df.set_index(join_on)["new_oai_mod"].apply(lambda d: d["id"]).sort_index()
            )
        ), "new_oai_mod id's off"
        assert (
            result_df.set_index(join_on)["new_oai_mod"]
            .apply(lambda d: chat_max_scores(d["results"]))[some_mod.values]
            .sort_index()
            .equals(
                merged_df.set_index(join_on)["new_oai_mod"]
                .apply(lambda d: chat_max_scores(d["results"]))
                .sort_index()
            )
        ), "oai_mod max scores off"
    return merged_df


# %%
chat_df = pd.read_pickle(f"data_dump/oai_mod/comparison_base_df18bd574.pkl")
analysis_mod_df = pd.read_pickle("data_dump/oai_mod/comp_results_ba0cefe.pkl")
merged_df = munge_check_mod_df(analysis_mod_df, chat_df)
# old_merged_df = pd.read_pickle("data_dump/oai_mod/merged_df_8b6b0fe.pkl") #1.7GB since of cosine sim
# assert old_merged_df[merged_df.columns].equals(merged_df)

chat_df2 = pd.read_pickle(f"data_dump/oai_mod/comparison_base_df53ca02c_lang_checks.pkl")
analysis_mod_df2 = pd.read_pickle("data_dump/oai_mod/comp_results_d7a8db3_lang_check.pkl")
merged_df2 = munge_check_mod_df(analysis_mod_df2, chat_df2)
# %%
merged_df["new_embedding"] = make_async_reqs(
    merged_df, max_workers=10, fn=lambda r: get_embeddings(r["new_sent_convo"]).data[0].embedding
)
merged_df["default_embedding"] = make_async_reqs(
    merged_df,
    max_workers=10,
    fn=lambda r: get_embeddings(r["default_sent_convo"]).data[0].embedding,
)
merged_df2["new_embedding"] = make_async_reqs(
    merged_df2, max_workers=10, fn=lambda r: get_embeddings(r["new_sent_convo"]).data[0].embedding
)
merged_df2["default_embedding"] = make_async_reqs(
    merged_df2,
    max_workers=10,
    fn=lambda r: get_embeddings(r["default_sent_convo"]).data[0].embedding,
)

merged_df["new_default_cos_dist"] = merged_df.apply(
    lambda r: cosine(r["new_embedding"], r["default_embedding"]), axis=1
)
merged_df2["new_default_cos_dist"] = merged_df2.apply(
    lambda r: cosine(r["new_embedding"], r["default_embedding"]), axis=1
)

# merged_df = pd.read_pickle("data_dump/oai_mod/merged_df_8b6b0fe.pkl") #1.7GB since of cosine sim
# merged_df2 = pd.read_pickle("data_dump/oai_mod/merged_df_8b6b0fe_lang_checks.pkl") # data_dump/oai_mod/merged_df_8b6b0fe_lang_checks.pkl
# merged_df = merged_df.rename(columns={"new_default_cos_sim": "new_default_cos_dist"})
# merged_df2 = merged_df2.rename(columns={"new_default_cos_sim": "new_default_cos_dist"})
merged_df.to_pickle(f"data_dump/oai_mod/merged_df_{git_hash()}.pkl")
merged_df2.to_pickle(f"data_dump/oai_mod/merged_df_{git_hash()}_lang_checks.pkl")


# %%
def mod_print_summaries(merged_df):
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
        "Percent of turns that got flagged that wouldn't have:"
        f" {100*np.mean(merged_df['new_any_flagged'] - merged_df['default_any_flagged'] > 0):.1f}%"
    )
    print(
        "Percent of turns with score diff 10ppt higher from sep:"
        f" {100*np.mean(merged_df['new_minus_default_max_score'] >= 0.1):.1f}%"
    )
    # sep helped
    print(
        "Percent of turns with score diff 10ppt lower from sep:"
        f" {100*np.mean(merged_df['new_minus_default_max_score'] <= -0.1):.1f}%"
    )
    missed_flag_ix = merged_df["new_any_flagged"] - merged_df["default_any_flagged"] < 0
    print(
        "Percent of turns that should've got flagged but didn't:"
        f" {100*np.mean(missed_flag_ix):.1f}%"
    )


# merged_df =pd.read_pickle("data_dump/oai_mod/merged_df_8a70b20.pkl")
# merged_df2 =pd.read_pickle("data_dump/oai_mod/merged_df_8a70b20_lang_checks.pkl")
# mod_print_summaries(merged_df)
# mod_print_summaries(merged_df2)

# %% Make all embeddings for merged_df3
# WARN: Requests
chat_df3 = pd.read_pickle("data_dump/oai_mod/_temp_comparison_base_df8a70b20_base3.pkl")
check_mod_df3 = pd.concat([
    make_results_frame(chat_df3, ord_vals=[192, None], model="text-moderation-latest"),
])
check_mod_df3 = cut(check_mod_df3)
# only running for convience, most mod scores will be discarded
check_mod_df3["new_oai_mod"] = make_async_reqs(check_mod_df3, max_workers=4)
check_mod_df3.to_pickle(f"data_dump/oai_mod/_tempcomp_results_{git_hash()}_base3.pkl")
analysis_mod_df3 = check_mod_df3
merged_df3 = munge_check_mod_df(analysis_mod_df3, chat_df3)

merged_df3["new_embedding"] = make_async_reqs(
    merged_df3, max_workers=10, fn=lambda r: get_embeddings(r["new_sent_convo"]).data[0].embedding
)
merged_df3["default_embedding"] = make_async_reqs(
    merged_df3,
    max_workers=10,
    fn=lambda r: get_embeddings(r["default_sent_convo"]).data[0].embedding,
)
merged_df3["new_default_cos_dist"] = merged_df3.apply(
    lambda r: cosine(r["new_embedding"], r["default_embedding"]), axis=1
)

merged_df3["new_embedding_small"] = make_async_reqs(
    merged_df3,
    max_workers=10,
    fn=lambda r: get_embeddings(r["new_sent_convo"], embeding_model="text-embedding-3-small")
    .data[0]
    .embedding,
)
merged_df3["default_embedding_small"] = make_async_reqs(
    merged_df3,
    max_workers=10,
    fn=lambda r: get_embeddings(r["default_sent_convo"], embeding_model="text-embedding-3-small")
    .data[0]
    .embedding,
)
merged_df3["new_default_cos_dist_small"] = merged_df3.apply(
    lambda r: cosine(r["new_embedding_small"], r["default_embedding_small"]), axis=1
)
merged_df3.to_pickle(f"data_dump/oai_mod/_temp_merged_df_{git_hash()}_base3.pkl")

merged_df3["new_embedding_ada002"] = make_async_reqs(
    merged_df3,
    max_workers=10,
    fn=lambda r: get_embeddings(r["new_sent_convo"], embeding_model="text-embedding-ada-002")
    .data[0]
    .embedding,
)
merged_df3["default_embedding_ada002"] = make_async_reqs(
    merged_df3,
    max_workers=10,
    fn=lambda r: get_embeddings(r["default_sent_convo"], embeding_model="text-embedding-ada-002")
    .data[0]
    .embedding,
)
merged_df3["new_default_cos_dist_ada002"] = merged_df3.apply(
    lambda r: cosine(r["new_embedding_ada002"], r["default_embedding_ada002"]), axis=1
)

merged_df3.to_pickle(f"data_dump/oai_mod/_temp_merged_df_{git_hash()}_base3_pt2.pkl")
# merged_df3 = pd.read_pickle("data_dump/oai_mod/_temp_merged_df_fac53a6_base3_pt2.pkl")

n = round(len(merged_df3) * 0.1)
top_10_percent = merged_df3.nlargest(n, "new_default_cos_dist")
print("top 10% of turns most changed by cos dist by text-embedding-3-large")
mod_print_summaries(top_10_percent)
top_10_percent.to_pickle(f"data_dump/oai_mod/merged_df_{git_hash()}_base3.pkl")

print("top 10% of turns most changed by cos dist by text-embedding-3-small")
top_10_percent_small = merged_df3.nlargest(n, "new_default_cos_dist_small")
mod_print_summaries(top_10_percent_small)
top_10_percent_small.to_pickle(f"data_dump/oai_mod/merged_df_{git_hash()}_base3_small.pkl")


# %%
# make a table of false positive and negative rates for each embedding model and per cutoff
def _get_top(df, col, per):
    n = round(len(df) * per)
    return df.nlargest(n, col)


def _fp_rate(df, col, per):
    df = _get_top(df, col, per)
    return 100 * np.mean(df["new_any_flagged"] & ~df["default_any_flagged"])


def _fn_rate(df, col, per):
    df = _get_top(df, col, per)
    return 100 * np.mean(~df["new_any_flagged"] & df["default_any_flagged"])


def print_3d_table_by_cols_name_fn(
    merged_df3,
    cutoffs=[0.25, 0.1, 0.05, 0.01],
    model_cols=(
        ("text-embedding-3-large", "new_default_cos_dist"),
        ("text-embedding-3-small", "new_default_cos_dist_small"),
        ("text-embedding-ada-002", "new_default_cos_dist_ada002"),
    ),
    name_fn=(
        ("false negative", _fn_rate),
        ("false positive", _fp_rate),
        ("net new unflagged", lambda df, col, per: _fn_rate(df, col, per) - _fp_rate(df, col, per)),
    ),
    print_sum=False,
):
    results = []
    for per in cutoffs:
        if print_sum:
            total_num_tokens = merged_df3["new_sent_convo"].apply(num_tokens_from_messages).sum()
            for model, col in model_cols:
                df = _get_top(merged_df3, col, per)
                token_frac = (
                    df["new_sent_convo"].apply(num_tokens_from_messages).sum() / total_num_tokens
                )
                print(
                    f"top {int(per*100)}% of turns by {model} cos dist is {token_frac*100:.2f}% of"
                    " tokens"
                )
                mod_print_summaries(df)
            continue
        # Compute the false positive and negative rates for each column by cutoff
        data = {
            (model, name): fn(merged_df3, col, per)
            for model, col in model_cols
            for name, fn in name_fn
        }

        data["cutoff"] = f"top {int(per*100)}%"
        # Store the results
        results.append(data)
    if print_sum:
        return
    df = pd.DataFrame(results)
    df.set_index("cutoff", inplace=True)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Model", "Metric"])
    # print df with floats  formated to 2 places
    with pd.option_context(
        "display.float_format",
        "{:,.1f}%".format,
    ):
        print(df)


# print_3d_table_by_cols_name_fn(merged_df3)
# print("\nEnglish Only\n")
# print_3d_table_by_cols_name_fn(merged_df3.query("language=='English'"))

# #print_3d_table_by_cols_name_fn(merged_df3, print_sum=True)
# #for 3-large 10% was 14% of tokens, 5% was 5.6% of tokens and 1% was 0.2% of tokens for whole df
# %% Making Big bad finetune dataset
check_fname = "data_dump/oai_mod/big_bad_finetune_check_mod_df4_fa6dc40.pkl"
if os.path.exists(check_fname):
    check_mod_df4 = pd.read_pickle(check_fname)
else:
    check_mod_df4 = make_results_frame(
        chat_df4,
        ord_vals=[192, None],
        model="text-moderation-latest",
        keep_old_oai_mod=True,
    )
    check_mod_df4 = cut(
        check_mod_df4
    )  # , strip_first_assistant=False) # no diff if don't strip assistants
    check_mod_df4.to_pickle(f"data_dump/oai_mod/big_bad_finetune_check_mod_df4_{git_hash()}.pkl")

preembed_fname = "data_dump/oai_mod/big_bad_finetune_preembed_df4_fa6dc40.pkl"
if os.path.exists(preembed_fname):
    preembed_df4 = pd.read_pickle(preembed_fname)
else:
    is_mod = check_mod_df4["manipulation"].apply(lambda d: d["sep"] is not None)
    preembed_df4 = check_mod_df4[is_mod].copy().rename(columns={"sent_convo": "new_sent_convo"})
    preembed_df4 = preembed_df4.join(
        check_mod_df4.loc[~is_mod, "sent_convo"].rename("default_sent_convo")
    )
    cid2lang = chat_df4[["conversation_id", "language"]].set_index("conversation_id")["language"]
    preembed_df4["language"] = preembed_df4["conversation_id"].apply(lambda c: cid2lang[c])
    print(preembed_df4["new_sent_convo"].apply(num_tokens_from_messages).describe())

    preembed_df4 = _split_convo_into_sep_turn_rows(
        preembed_df4,
        cols2split=["new_sent_convo", "default_sent_convo", "og_openai_moderation"],
        cols_as_list=["new_sent_convo", "default_sent_convo", "og_openai_moderation"],
        run_asserts=False,
    )
    # Don't make dup free, since want these to send as part of whole convos
    preembed_df4.to_pickle(f"data_dump/oai_mod/big_bad_finetune_preembed_df4_{git_hash()}.pkl")


# %% TODO
def make_api_requests_in_batches(
    in_df,
    model,
    token_encoding_name="cl100k_base",
    max_attempts=6,
    logging_level=20,  # logging.info
    api_key=api_key,
):
    if len(in_df) == 0:
        return in_df
    assert isinstance(in_df["input"].iloc[0], str), in_df.iloc[0]
    endpoint = get_endpoint(model)
    max_requests_per_minute, max_tokens_per_minute, request_url = {
        "embedding": (10000, 10000000, "https://api.openai.com/v1/embeddings"),
        "moderation": (1000, 150000, "https://api.openai.com/v1/moderations"),
    }[endpoint]

    dir = f"data_dump/oai_mod/temp_throughput"
    if not os.path.exists(dir):
        os.mkdir(dir)
    formatted_now = datetime.now().strftime("%m_%d_%H_%M_%S_%f")
    requests_filepath = f"{dir}/req_{endpoint}_{formatted_now}.jsonl"
    save_filepath = f"{dir}/req_{endpoint}_{formatted_now}_result.jsonl"

    # {"model": "text-embedding-3-small", "input": "embed me", "metadata": {"row_id": 1}}

    with open(requests_filepath, "w") as f:

        for iloc, (loc, row) in enumerate(in_df.iterrows()):
            data = {
                "model": model,
                "input": row["input"],
                "metadata": {
                    "loc": loc,
                    "iloc": iloc,
                    **(row["metadata"] if "metadata" in row else {}),
                },
            }
            f.write(json.dumps(data) + "\n")

    asyncio.run(
        process_api_requests_from_file(
            requests_filepath,
            save_filepath,
            request_url,
            api_key,
            max_requests_per_minute,
            max_tokens_per_minute,
            token_encoding_name,
            max_attempts,
            logging_level,
        )
    )
    rows = []

    # Open the JSONL file and read it line by line
    with open(save_filepath, "r") as f:
        for line in f:
            # Parse the JSON line into a Python object (list of dictionaries)
            data = json.loads(line)
            if endpoint == "embedding":
                obj_data = data[1]["data"]
            elif endpoint == "moderation":
                obj_data = data[1]  # no .model_dump() since reading from file
            record = {
                "model_input": data[0],
                "object_data": obj_data,  # Keeping the second dict as is
                "metadata": data[2],
            }
            rows.append(record)
    result_df = pd.DataFrame(rows)

    # If needed, further processing to expand dicts into separate columns
    result_df = result_df.join(pd.json_normalize(result_df["model_input"])).drop(
        columns=["model_input"]
    )

    result_df = result_df.join(pd.json_normalize(result_df["metadata"])).drop(columns=["metadata"])
    result_df = result_df.sort_values(by="iloc").set_index("loc").drop(columns="iloc")
    return result_df


# %%
# Warn: Requests
import asyncio
import nest_asyncio
import importlib
import src.api_request_parallel_processor
import heapq

importlib.reload(src.api_request_parallel_processor)
from src.api_request_parallel_processor import process_api_requests_from_file


nest_asyncio.apply()


def get_endpoint(model):
    if "embedding" in model:
        endpoint = "embedding"
    elif "moderation" in model:
        endpoint = "moderation"
    else:
        assert False, model
    return endpoint


def make_api_requests(
    in_df,
    model,
    token_encoding_name="cl100k_base",
    max_attempts=6,
    logging_level=20,  # logging.info
    api_key=api_key,
):
    if len(in_df) == 0:
        return in_df
    assert isinstance(in_df["input"].iloc[0], str), in_df.iloc[0]
    endpoint = get_endpoint(model)
    max_requests_per_minute, max_tokens_per_minute, request_url = {
        "embedding": (10000, 10000000, "https://api.openai.com/v1/embeddings"),
        "moderation": (1000, 150000, "https://api.openai.com/v1/moderations"),
    }[endpoint]

    dir = f"data_dump/oai_mod/temp_throughput"
    if not os.path.exists(dir):
        os.mkdir(dir)
    formatted_now = datetime.now().strftime("%m_%d_%H_%M_%S_%f")
    requests_filepath = f"{dir}/req_{endpoint}_{formatted_now}.jsonl"
    save_filepath = f"{dir}/req_{endpoint}_{formatted_now}_result.jsonl"

    # {"model": "text-embedding-3-small", "input": "embed me", "metadata": {"row_id": 1}}

    with open(requests_filepath, "w") as f:
        for iloc, (loc, row) in enumerate(in_df.iterrows()):
            data = {
                "model": model,
                "input": row["input"],
                "metadata": {
                    "loc": loc,
                    "iloc": iloc,
                    **(row["metadata"] if "metadata" in row else {}),
                },
            }
            f.write(json.dumps(data) + "\n")

    asyncio.run(
        process_api_requests_from_file(
            requests_filepath,
            save_filepath,
            request_url,
            api_key,
            max_requests_per_minute,
            max_tokens_per_minute,
            token_encoding_name,
            max_attempts,
            logging_level,
        )
    )
    rows = []

    # Open the JSONL file and read it line by line
    with open(save_filepath, "r") as f:
        for line in f:
            # Parse the JSON line into a Python object (list of dictionaries)
            data = json.loads(line)
            if endpoint == "embedding":
                obj_data = data[1]["data"]
            elif endpoint == "moderation":
                obj_data = data[1]  # no .model_dump() since reading from file
            record = {
                "model_input": data[0],
                "object_data": obj_data,  # Keeping the second dict as is
                "metadata": data[2],
            }
            rows.append(record)
    result_df = pd.DataFrame(rows)

    # If needed, further processing to expand dicts into separate columns
    result_df = result_df.join(pd.json_normalize(result_df["model_input"])).drop(
        columns=["model_input"]
    )

    result_df = result_df.join(pd.json_normalize(result_df["metadata"])).drop(columns=["metadata"])
    result_df = result_df.sort_values(by="iloc").set_index("loc").drop(columns="iloc")
    return result_df


def make_df_requests(df, input_col, model, result_col=None):
    """
    High Throughput api_requests,
    res_col specifies which data to reuse, if exists
    """
    if result_col not in df:
        print(f"INFO: {result_col} not yet in df, making form scratch")
        result_col = None
    if result_col is None:
        ix_already_have = np.array([False] * len(df))
    else:
        ix_already_have = ~df[result_col].isnull()
        if ix_already_have.mean() == 1:
            print(f"INFO: Already have all entries for {result_col}")
            return df[result_col]

    send_api_df = (
        df[input_col][~ix_already_have]
        .apply(lambda l: l[0]["content"] if len(l[0]["content"]) else " ")
        .to_frame()
    )
    send_api_df.columns = ["input"]
    print(f"starting {datetime.now()} {model} on input {input_col} making {result_col} col")
    result_api_df = make_api_requests(
        send_api_df,
        model,
        token_encoding_name="cl100k_base",
        max_attempts=6,
        logging_level=20,
    )
    print(f"finished {datetime.now()} {model} on input {input_col} making {result_col} col")
    assert result_api_df.index.equals(send_api_df.index)
    assert result_api_df["input"].equals(send_api_df["input"])

    endpoint = get_endpoint(model)
    if endpoint == "moderation":
        out_data = result_api_df["object_data"]
        # now a list of dicts
        assert out_data.apply(lambda l: len(l["results"]) == 1).all(), result_api_df
    elif endpoint == "embedding":
        assert result_api_df["object_data"].apply(lambda l: len(l) == 1).all()
        out_data = result_api_df["object_data"].apply(lambda l: np.array(l[0]["embedding"]))
        # res_series = res_series.apply(lambda res: res.data)
        # out_data = res_series.apply(lambda res: np.array(res[0].embedding))

    if result_col is None:
        assert out_data.index.equals(df.index)
        return out_data
    return df[result_col].where(ix_already_have, out_data).copy()


def get_largest_cosine(file1, file2, df, per_keep, out_col_name):
    """
    From 2 jsonl files, each with lines of [[{"model": "", "input":""}, {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [999]}],"model": "text-embedding-3-small", "usage": {"prompt_tokens": 803, "total_tokens": 803}}, {"loc": 1514, "iloc": 0}]]
    calculate the cosine distance between the embeddings where each iloc lines up.
    The iloc are mostly, but not perfectly sorted by iloc
    The jsonl files are too large to be loaded into memory directly
    return the top per_keep percent of ilocs on the df with the highest cosine distance between the two files.
    """
    f1_ilocs = {}
    f2_ilocs = {}
    top_n_ilocs = round(len(df) * per_keep)
    pq = []

    def _add(dist, iloc):
        """
        keeps the biggest values seen.
        It's a min heap so smallest values are popped off root after length
        """
        assert dist >= 0
        if len(pq) < top_n_ilocs:
            heapq.heappush(pq, (dist, iloc))
        else:
            heapq.heappushpop(pq, (dist, iloc))

    def process_ilocs(ilocs_dict, iloc, emb, other_dict, _add):
        if iloc in other_dict:
            cos_dist = cosine(emb, other_dict[iloc])
            _add(cos_dist, iloc)
            del other_dict[iloc]
        else:
            ilocs_dict[iloc] = emb

    with open(file1, "r") as f1, open(file2, "r") as f2:
        for ix, (line1, line2) in enumerate(zip(f1, f2)):
            data1 = json.loads(line1)
            data2 = json.loads(line2)

            emb1 = np.array(data1[1]["data"][0]["embedding"])
            emb2 = np.array(data2[1]["data"][0]["embedding"])
            iloc1 = data1[2]["iloc"]
            iloc2 = data2[2]["iloc"]
            if iloc1 == iloc2:
                cos_dist = cosine(emb1, emb2)
                _add(cos_dist, iloc1)
            else:
                process_ilocs(f1_ilocs, iloc1, emb1, f2_ilocs, _add)
                process_ilocs(f2_ilocs, iloc2, emb2, f1_ilocs, _add)
            if ix % (len(df) // 10) == 0:
                print(
                    f"{datetime.now()} {ix}/{len(df)} has"
                    f" {ix + 1 -(len(f1_ilocs) + len(f2_ilocs))/2} matched; but {len(f1_ilocs)} f1"
                    f" missing and {len(f2_ilocs)} f2 missing"
                )
    for iloc1, emb1 in f1_ilocs.items():
        process_ilocs(f1_ilocs, iloc1, emb1, f2_ilocs, _add)
    for iloc2, emb2 in f2_ilocs.items():
        process_ilocs(f2_ilocs, iloc2, emb2, f1_ilocs, _add)

    if len(f1_ilocs) > 0:
        print(f"WARN: Didn't match {len(f1_ilocs)} from file 1 {file1}")
    if len(f2_ilocs) > 0:
        print(f"WARN: Didn't match {len(f2_ilocs)} from file 2 {file2}")
    if len(pq) < top_n_ilocs:
        print(f"WARN: Only have {len(pq)} vs {top_n_ilocs} expected")

    out = df.iloc[[ix for _, ix in pq], :].copy()
    out[out_col_name] = np.array([cos_dist for cos_dist, _ in pq])
    out.sort_values(by=out_col_name, ascending=False, inplace=True)
    return out


PER_KEEP = 0.05
if len(preembed_df4) < 100000:
    # # embedding cols are 250k embeddings * 1536 floats/embeddings * 4 bytes/float = 1.5GB
    pass
    # preembed_df4["new_embedding_small"] = make_batch_requests(
    #     preembed_df4, "new_sent_convo", "text-embedding-3-small"
    # )
    # preembed_df4["default_embedding_small"] = make_batch_requests(
    #     preembed_df4, "default_sent_convo", "text-embedding-3-small"
    # )
    # preembed_df4["new_default_cos_dist_small"] = preembed_df4.apply(
    #     lambda r: cosine(r["new_embedding_small"], r["default_embedding_small"]), axis=1
    # )
    # preembed_df4.to_pickle(f"data_dump/oai_mod/big_bad_finetune_added_embeddings_df4_{git_hash()}.pkl")

    # n = round(len(preembed_df4) * PER_KEEP)
    # big_dist_df4 = preembed_df4.nlargest(n, "new_default_cos_dist_small")
else:
    # make_batch_requests(
    #     preembed_df4, "new_sent_convo", "text-embedding-3-small"
    # )
    #
    # make_batch_requests(
    #     preembed_df4, "default_sent_convo", "text-embedding-3-small"
    # )
    big_dist_df4 = get_largest_cosine(
        # new_sent_convo small missing 3 rows
        "data_dump/oai_mod/temp_throughput/req_embedding_02_12_18_11_09_968134_result_not_quite_finished.jsonl",
        # default_sent_convo small
        "data_dump/oai_mod/temp_throughput/req_embedding_02_12_21_11_30_511517_result_not_quite_finished.jsonl",
        preembed_df4,
        PER_KEEP,
        "new_default_cos_dist_small",
    )
    # matched all but 3
    big_dist_df4.to_pickle(f"data_dump/oai_mod/big_bad_finetune_filtered_cos_{git_hash()}.pkl")

big_dist_df4["new_oai_mod"] = make_df_requests(
    big_dist_df4, "new_sent_convo", "text-moderation-latest"
)
big_dist_df4["new_max_score"] = big_dist_df4["new_oai_mod"].apply(chat_max_scores)
big_dist_df4["new_any_flagged"] = big_dist_df4["new_oai_mod"].apply(chat_is_flagged)
big_dist_df4["og_max_score"] = big_dist_df4["og_openai_moderation"].apply(chat_max_scores)
big_dist_df4["og_any_flagged"] = big_dist_df4["og_openai_moderation"].apply(chat_is_flagged)
big_dist_df4.to_pickle(f"data_dump/oai_mod/big_bad_finetune_w_mod_{git_hash()}.pkl")

df4 = big_dist_df4.rename(
    columns={"og_any_flagged": "default_any_flagged", "og_max_score": "default_max_score"}
)

print_3d_table_by_cols_name_fn(
    df4,
    cutoffs=[1, 0.5, 0.25, 0.1, 0.05, 0.01],
    model_cols=(("text-embedding-3-small", "new_default_cos_dist_small"),),
)
#%%
# # format: {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
# each turn has 4k, limit should already be short enough
from src.make_prompts import make_prompt13


def single_turn(only_one, lang, system=[]):
    assert len(only_one) == 1, only_one
    if isinstance(only_one, pd.Series):
        only_one = only_one.iloc[0]
    role, content = only_one[0]["role"], only_one[0]["role"]
    if num_tokens_from_string(content) < 5:
        return False
    start_tokens = take_first_tokens(content, 5)
    if role == "user":
        return {
            "messages": system
            + [{"role": "user", "content": f"In {lang} continue {start_tokens}"}]
            + [{"role": "assistant", "content": content}]
        }
    elif role == "assistant":
        return {"messages": system + only_one}
    else:
        assert False


def filter_and_join(group):
    # Filter the messages where 'new_any_flagged' is false
    filtered_group = group[group["new_any_flagged"] == False]
    assert len(filtered_group) > 0, group
    lang = filtered_group["language"].iloc[0]
    sep = filtered_group["manipulation"].iloc[0]["sep"]
    system = [{"role": "system", "content": make_prompt13(sep)}]
    if len(filtered_group) == 1:
        only_one = filtered_group["new_sent_convo"]
        return single_turn(only_one, lang, system)

    # not needed?
    # # Ensure that the 'role' alternates between users and assistant
    # roles = filtered_group["new_sent_convo"].apply(lambda l: l[0]["role"])
    # filtered_group = filtered_group[roles != roles.shift()]
    # if len(filtered_group) <= 1:
    #     return single_turn(filtered_group["new_sent_convo"], lang, system)

    # Join the messages into a single string
    convo_joined = sum(filtered_group["new_sent_convo"].tolist(), [])
    if all((c['role']=='user' for c in convo_joined)):
        convo_joined = [{'role':'assistant', 'content':c['content']} for c in convo_joined]
        #n_content = convo_joined[0]['content']
        #convo_joined.insert(0, {"role":"user","content":f"Tell me a story that continues: `{n_content}`"})
    return {"messages": system + convo_joined}


# Apply the function to each group
def make_finetune_convos1(df, convo_ids=None):
    if convo_ids is None:
        convo_ids = df.query("default_any_flagged ==True and new_any_flagged ==False")[
            "conversation_id"
        ].unique()
    df_grouped = (
        df[df["conversation_id"].isin(convo_ids)]
        .groupby("conversation_id")
        .apply(filter_and_join)
        .reset_index()
        .rename(columns={0: "finetune_convo"})
        .query("finetune_convo!=False")
        .reset_index()
    )
    return df_grouped


def start_finetune(send_df, fname):
    """Convert a df into a json file and start a finetune training"""
    assert len(send_df) > 10, "min finetune len is 10 convos"

    dir = "data_dump/oai_mod/finetune"
    if not os.path.exists(dir):
        os.mkdir(dir)
    filepath = f"{dir}/{fname}_{git_hash()}.json"
    send_df.to_pickle(filepath.replace(".json", ".pkl"))
    send_df["finetune_convo"].to_json(filepath, orient="records", lines=True, index=False)

    fres = client.files.create(file=open(filepath, "rb"), purpose="fine-tune")
    print(fres)
    return client.fine_tuning.jobs.create(
        training_file=fres.id,
        # model="gpt-4-0613", #code DNE but docs says exist
        model="gpt-3.5-turbo-1106",
        hyperparameters={
            "n_epochs": 1,
        },
    )


# n = round(len(merged_df3) * 0.5)
# df3 = merged_df3.nlargest(n, "new_default_cos_dist")
# start_finetune(df3, "chat_df3_test.json")
# # ft:gpt-3.5-turbo-1106:personal::8q9ZY1NX
# # Succesfully trained
# # First Proof you can finetune where was flagged but now isn't


# data all made
# use original old oai endpoint for filtering
print(
    "Total bad Token through 2 steps: ",
    df4["default_sent_convo"].apply(num_tokens_from_messages).describe(),
)
sending_df4 = df4.query("new_any_flagged==False and new_max_score < 0.55")
print(
    "Total bad Token through filtering: ",
    sending_df4["default_sent_convo"].apply(num_tokens_from_messages).describe(),
)

made_convo = make_finetune_convos1(df4, convo_ids=sending_df4.conversation_id.unique())
print(
    "Tokenings sending to finetune: ",
    made_convo["finetune_convo"]
    .apply(lambda d: num_tokens_from_messages(d["messages"]))
    .describe(),
)
start_finetune(made_convo, "chat_df4_ft1.json")


# %%
# Read the strings where adding seperators worked
def _write_where_missed_flagging(merged_df):
    missed_flag_ix = merged_df["new_any_flagged"] - merged_df["default_any_flagged"] < 0
    not_flagged_convos = merged_df[missed_flag_ix]["default_sent_convo"]
    print(not_flagged_convos.value_counts().value_counts().sort_index())

    value_counts = not_flagged_convos.value_counts()
    df = value_counts[value_counts > len(ORD_USE_BETWEEN) / 2].reset_index()
    df.columns = ["index", "value_counts"]
    df = df[["value_counts", "index"]]
    df.to_csv(f"data_dump/oai_mod/mostly_passed_flagging_{git_hash()}.csv", sep="\t", index=False)


# _write_where_missed_flagging(merged_df)
# 133 strs only flagged by 1 sep, then ~30-40 work for 2-7 sep's
# %%
# compare the langauge where adding serperators mostly worked
def print_missed_flag_by_lang_analysis(merged_df, print_only_sig=True, pval=0.001):
    missed_flag_ix = merged_df["new_any_flagged"] - merged_df["default_any_flagged"] < 0
    lang_default = merged_df["language"].value_counts()
    lang_missed_flag = merged_df["language"][missed_flag_ix].value_counts()
    exp_lang_missed_flag = merged_df["language"].value_counts(normalize=True) * missed_flag_ix.sum()

    results = {}
    p = missed_flag_ix.sum() / len(merged_df)  # Success probability under null hypothesis
    for language in lang_default.index:
        n = lang_default.loc[language]  # Number of trials
        k = lang_missed_flag.get(language, 0)  # Number of successes
        # Binomial test
        p_value = binomtest(k, n, p).pvalue
        results[language] = {"p_value": p_value, "ratio_change": (k / n) / p}

    sig_langs = {lan: d for lan, d in results.items() if d["p_value"] < pval / len(lang_default)}
    # only sig or full
    if print_only_sig:
        if len(sig_langs) == 0:
            print(f"No Significant langs at {pval/len(lang_default)}")
            pprint(results)
            return
        sig_langs_df = pd.DataFrame(sig_langs.values(), index=sig_langs.keys())
    else:
        sig_langs_df = pd.DataFrame(results.values(), index=results.keys())
        print("INFO: printing all languages")
    sig_langs_df["num_missed_flagged"] = lang_missed_flag[sig_langs_df.index]
    sig_langs_df["exp_num_missed_flagged"] = (
        exp_lang_missed_flag[sig_langs_df.index].round(0).astype(int)
    )
    sig_langs_df["per_of_flags_missed"] = (
        lang_missed_flag[sig_langs_df.index] / lang_default[sig_langs_df.index]
    )
    with pd.option_context(
        "display.float_format",
        "{:,.2e}".format,
        "display.max_columns",
        None,
        "display.expand_frame_repr",
        False,
    ):
        sig_langs_df["per_of_flags_missed"] = sig_langs_df["per_of_flags_missed"].apply(
            lambda x: "{:.1%}".format(x)
        )
        sig_langs_df["ratio_change"] = sig_langs_df["ratio_change"].apply(
            lambda x: "{:.2}".format(x)
        )
        print(sig_langs_df.sort_values("p_value"))
    print(
        lang_default.loc[sig_langs.keys()] / lang_default.sum(),
        lang_missed_flag.loc[sig_langs.keys()] / lang_missed_flag.sum(),
    )


print_missed_flag_by_lang_analysis(merged_df, print_only_sig=True)

sep_is_192 = merged_df2["manipulation"].apply(lambda d: d["sep"] == chr(192))
# spliting out don't see a real difference
print_missed_flag_by_lang_analysis(merged_df2[~sep_is_192], print_only_sig=False)
print_missed_flag_by_lang_analysis(merged_df2[sep_is_192], print_only_sig=False)
print_missed_flag_by_lang_analysis(merged_df2, print_only_sig=False)

print_missed_flag_by_lang_analysis(top_10_percent, print_only_sig=True)
# %%
# Make Plots
from scipy.stats import ks_2samp
import scipy.stats as stats
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


def _ks_hist_plot(data1, data2, name1=None, name2=None, ax=None, sig_level=0.05):
    if ax is None:
        fig, ax = plt.subplots()

    name1, name2 = get_name(data1, name1, "1"), get_name(data2, name2, "2")
    ax.hist(
        data1,
        color=str_to_color(name1),
        alpha=0.5,
        label=name1 + f" m: {data1.mean():.3f} sem: {data1.sem():.3f}",
    )
    ax.hist(
        data2,
        color=str_to_color(name2),
        alpha=0.5,
        label=name2 + f" m: {data2.mean():.3f} sem: {data2.sem():.3f}",
    )
    statistic, p_value = ks_2samp(data1.dropna(), data2.dropna(), alternative="two-sided")
    title = f"{name1} vs {name2}"
    title += f"\nKS Statistic: {statistic:.3f}, P-Value: {p_value:.3f}"
    color = "red" if p_value < sig_level else "black"
    ax.set_title(title, color=color)
    ax.legend()
    return ax


def Sent_make_summary_hist_plot(merged_df, name):
    data1 = merged_df["new_max_score"]
    data2 = merged_df["default_max_score"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = _ks_hist_plot(data1, data2, name1="with seperators", name2="w/o seperators", ax=ax)
    fig.suptitle(f"{', '.join(merged_df['mod_model'].unique())} Max Category Score per Message")
    fig.subplots_adjust(top=0.86)
    fig.savefig(
        f"plots/oai_mod/average_max_scores_yn_seperators_{git_hash()}_{name}.png", facecolor="w"
    )


def reg_plot(x1, y1, x_name=None, y_name=None, title=None, ax=None):
    x_name, y_name = get_name(x1, x_name, "X"), get_name(y1, y_name, "Y")
    if title is None:
        title = f"{y_name} vs {x_name}"

    ax = sns.regplot(
        x=x1, y=y1, scatter=True, ci=95, line_kws={"color": "red"}, scatter_kws={"s": 2}, ax=ax
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
        [f"{interval.mid:.2f}" for interval in bucket_means.index],
        rotation=90,
    )
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.figure.tight_layout()
    return ax


def Show_prompt_lengths_vs_max_score(df, by_width=False):
    """
    plot both prompt length and output length vs max mod score
    """
    prompt_lens = df["new_sent_convo"].apply(num_tokens_from_messages)
    prompt_lens.name = "Sent Convo Num Tokens"
    og_prompt_lens = df["default_sent_convo"].apply(num_tokens_from_messages)
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
    by_str = "num_tokens" if by_width else "num_tokens quantile"
    ax.set_title(f"Sent Prompt Lengths vs Score Difference grouped by {by_str} ")
    plt.show()

    ax = avg_by_bucket(og_prompt_lens, score_diff, by_width=by_width)
    ax.set_title(f"Original Prompt Lengths vs Score Difference grouped by {by_str} ")
    plt.show()


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
                _ks_hist_plot(data1, data2, name1=cat1, name2=cat2, ax=ax, sig_level=sig_level)
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
def Sent_by_cat_sep_vs_sep(merged_df, name):
    path = f"plots/oai_mod/compare_sep_tokens_on_oai_mod_{git_hash()}"
    if not os.path.exists(path):
        os.mkdir(path)
    for c in list(merged_df["new_oai_mod"].iloc[0]["results"][0]["category_scores"].keys())[1:]:
        diff = merged_df[f"new_{c}"] - merged_df[f"default_{c}"]
        fig = plot_comparisons(merged_df, "mod_how_str", diff)
        fig.suptitle(
            f"Compare different preprocessing steps on difference in {c} numeric scores",
            fontsize=50,
        )
        fig.savefig(
            f"{path}/{name}_{c.replace('/', '')}.png",
            facecolor="w",
            bbox_inches="tight",
        )


# %%
Sent_make_summary_hist_plot(merged_df, "df1")
plt.show()
Show_prompt_lengths_vs_max_score(merged_df, by_width=False)
plt.show()
Sent_by_cat_sep_vs_sep(merged_df, "df1")

Sent_make_summary_hist_plot(merged_df2, "lang_proc")
plt.show()
Show_prompt_lengths_vs_max_score(merged_df2, "lang_proc", by_width=False)
plt.show()
Sent_by_cat_sep_vs_sep(merged_df2, "lang_proc")


# %%
# see if embeddings different
def plot_bucket_cosine_dist(merged_df, name, nrows=50):
    sorted_df = merged_df.sort_values(by="new_minus_default_max_score", key=np.abs)
    big_cos = sorted_df.iloc[-nrows:]["new_default_cos_dist"]
    small_cos = sorted_df.iloc[:nrows]["new_default_cos_dist"]

    fig, ax = plt.subplots(figsize=(10, 6))
    _ks_hist_plot(
        small_cos,
        big_cos,
        name1=f"cosine dist of {nrows} smallest abs(new - default)",
        name2=f"cosine dist of {nrows} largest abs(new - default)",
        ax=ax,
    )
    fig.suptitle(f"Does adding seperaters change embedding for {name}?")
    fig.subplots_adjust(top=0.86)
    fig.tight_layout()
    fig.savefig(f"plots/oai_mod/embedding_cos_dist_ks_hist_{git_hash()}_{name}.png", facecolor="w")
    fig.show()
    # plt.close(fig)


def reg_cosine_dist(
    merged_df,
    name,
    x_name="new_default_cos_dist",
    y_name="new_minus_default_max_score",
    embedding_model="text-embedding-3-large",
):
    fig, ax = plt.subplots(figsize=(10, 6))
    reg_plot(
        merged_df[x_name],
        merged_df[y_name],
        x_name="Embedding cosine distance between turn w/ and w/o sep token",
        y_name="Avg Max Mod with seperators - w/o seperators",
        title=f"Embeding by {embedding_model} vs Max Mod score difference for {name}",
    )
    fig.tight_layout()
    fig.savefig(f"plots/oai_mod/embedding_cos_dist_reg_{git_hash()}_{name}.png", facecolor="w")
    fig.show()


def show_reg_plot_cos_dist_by_group_avg(
    merged_df, name, x_name="new_default_cos_dist", y_name="new_minus_default_max_score"
):
    df = (
        merged_df.groupby(["language", "mod_how_str"])[[x_name, y_name]]
        .mean()
        .sort_values("new_default_cos_dist")
    )
    df.reset_index(inplace=True)
    plt.figure(figsize=(10, 8))
    sns.regplot(x=x_name, y=y_name, data=df, fit_reg=True)

    # Add labels
    for i in range(df.shape[0]):
        plt.text(
            df.loc[i, "new_default_cos_dist"],
            df.loc[i, "new_minus_default_max_score"],
            f"{df.loc[i, 'language']}-{df.loc[i, 'mod_how_str']}",
            horizontalalignment="left",
            size="medium" if len(df) < 20 else "small",
            color="black",
            weight="semibold" if len(df) < 10 else "light",
        )

    corr, p = stats.pearsonr(df[x_name], df[y_name])
    plt.text(
        0.75,
        1.1,
        f"corr: {corr:.2f} p: {p:.2f}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    plt.xlabel("Mean New Default Cosine Distance")
    plt.ylabel("Mean New Minus Default Max Score")
    plt.title(f"Regression Plot with Group Averages for {name}")
    plt.show()


# %%
plot_bucket_cosine_dist(merged_df, "df1", nrows=50)
reg_cosine_dist(merged_df, "df1")
show_reg_plot_cos_dist_by_group_avg(merged_df, "df1")

plot_bucket_cosine_dist(merged_df2, "lang_checks", nrows=50)
reg_cosine_dist(merged_df2, "lang_checks")
show_reg_plot_cos_dist_by_group_avg(merged_df2, "lang_checks")

reg_cosine_dist(
    merged_df3,
    "high_cosine_dist",
    x_name="new_default_cos_dist_small",
    embedding_model="text-embedding-3-small",
)
# %%
embedding_dist = pd.concat([df["new_default_cos_dist"] for df in (merged_df, merged_df2)])
score_diff = pd.concat([df["new_minus_default_max_score"] for df in (merged_df, merged_df2)])

ax = avg_by_bucket(embedding_dist, score_diff, by_width=True)  # xaxis 0 rounded by default
ax.set_title(f"Total embedding correlation distance vs score diff by bucket_size")
plt.show()

ax = avg_by_bucket(embedding_dist, score_diff, by_width=False)
ax.set_title(f"Total embedding correlation distance vs score diff by quantile")
plt.show()

print(
    "Average Cosine Distance by langauge\n",
    merged_df.groupby(["language"])[["new_default_cos_dist", "new_minus_default_max_score"]]
    .mean()
    .sort_values("new_default_cos_dist"),  # agg(['mean', 'sem']),
    merged_df2.groupby(["language"])[["new_default_cos_dist", "new_minus_default_max_score"]]
    .mean()
    .sort_values("new_default_cos_dist"),  # agg(['mean', 'sem']),
)

print(
    "Average Cosine Distance by langauge and mod_how_str",
    merged_df.groupby(["language", "mod_how_str"])[
        ["new_default_cos_dist", "new_minus_default_max_score"]
    ]
    .mean()
    .sort_values("new_default_cos_dist"),  # agg(['mean', 'sem']),
    merged_df2.groupby(["language", "mod_how_str"])[
        ["new_default_cos_dist", "new_minus_default_max_score"]
    ]
    .mean()
    .sort_values("new_default_cos_dist"),  # agg(['mean', 'sem']),
)
# %% Cutoff line for usefulness: code below was run but not used in report

# %%
# run logistic regression predict if will missidentify flag
# But basically on predicts 0
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def print_log_pred(X, Y, df, random_state=0, cutoffs=[1, 0.5, 0.25, 0.1, 0.05]):
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.33, random_state=random_state
    )
    # X_train, X_test, y_train, y_test = X, X, Y, Y
    class_weights = {1: 1, 0: 0.5, -1: 1}
    if True:
        clf = LogisticRegression(random_state=random_state, class_weight=class_weights)
        # clf = LogisticRegression(random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # y_pred = X_test.iloc[:, 1]
        # print(accuracy_score(y_test, y_pred))
        print(f"Fraction 0's pred: {(y_pred==0).mean()*100:.1f}%")
    else:
        clf = RandomForestClassifier(random_state=random_state, class_weight=class_weights)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))

    test_df = df.loc[X_test.index].copy()
    # not actually sorting on pred_net_unflagged?!?!
    test_df["pred_net_unflagged"] = y_pred
    # print(_fn_rate(test_df, "pred_net_unflagged", 0.1))

    for c in cutoffs:
        top_n_y_test = y_test[_get_top(test_df, "pred_net_unflagged", c).index]
        assert _fn_rate(test_df, "pred_net_unflagged", c) == np.mean(top_n_y_test > 0) * 100

    print_3d_table_by_cols_name_fn(
        test_df,
        print_sum=False,
        cutoffs=cutoffs,
        model_cols=(("predicted net unflagged", "pred_net_unflagged"),),
        name_fn=(
            ("false negative", _fn_rate),
            ("false positive", _fp_rate),
            (
                "net new unflagged",
                lambda df, col, per: _fn_rate(df, col, per) - _fp_rate(df, col, per),
            ),
        ),
    )


def pre_log_df(df):
    column_names = [
        "new_default_cos_dist",
        "new_default_cos_dist_small",
        "new_default_cos_dist_ada002",
    ]
    cos_dists = df[df.columns.intersection(column_names)]
    before = cos_dists.copy()
    before["og_prompt_len"] = df["default_sent_convo"].apply(num_tokens_from_messages)
    if "og_openai_moderation" in df:
        og_mod = df["og_openai_moderation"].apply(lambda d: pd.Series(d[0]["category_scores"]))
        before = pd.concat([before, og_mod], axis=1)
    # Want avg(Y) as high as possible: used to be now not flagged
    Y = df["default_any_flagged"] - df["new_any_flagged"]
    return before, Y


def _check_all(df):
    """
    Only pred's 0's if have to predict from full dataset.
    Tons of variance from which subset used
    """
    before, Y = pre_log_df(df)
    print_log_pred(before, Y, df, random_state=0)
    print_log_pred(before, Y, df, random_state=1)
    print_log_pred(before, Y, df, random_state=2)
    # WARN: There's a huge variance in how predictive the column is based on the subset used

    # These still don't help
    # print("INFO: With moderation scores that won't have in prod")
    # cat_max_score_diffs = df["new_oai_mod"].apply(
    #    lambda d: pd.Series(d["results"][0]["category_scores"])
    # ) - df["default_oai_mod"].apply(lambda d: pd.Series(d["results"][0]["category_scores"]))
    # X = pd.concat([cos_dists, cat_max_score_diffs], axis=1)
    # print_log_pred(X, Y, df)


def _check_part(in_df, cutoffs=[1, 0.5, 0.25, 0.1]):
    """
    pre-filter on old og_max_mod before sending to log reg
    Since all merge_df's have already been filtered once on og_openai_moderation
        this doesn't have an effect?
    With high cutoffs only things that are flagged by both,
        so total result is 1/0 fp and 0 fn.
    Only "things in the middle" that escape mod?
    """
    in_df = in_df.copy()
    in_df["og_max_mod"] = in_df["og_openai_moderation"].apply(chat_max_scores)
    for c in cutoffs:
        df = _get_top(in_df, "og_max_mod", c)
        before, Y = pre_log_df(df.drop("og_openai_moderation", axis=1))
        print(f"Prefiltered Cutoff at: {c}")
        print_log_pred(before, Y, df, random_state=2, cutoffs=[1, 0.5, 0.25, 0.1])


df3_wold = merged_df3.copy()
cid2oai = chat_df3[["conversation_id", "openai_moderation"]].set_index("conversation_id")
df3_wold["og_openai_moderation"] = df3_wold.apply(
    lambda r: [cid2oai.loc[r["conversation_id"]][0][r["convo_ix"]]], axis=1
)
# _check_all(df3_wold.query('language=="English"'))
# _check_part(df3_wold)  # .query('language=="English"'))
_check_all(merged_df2.query('language=="Russian"'))
_check_all(merged_df2)
# Russian does way better than everything combined
# %%
# Big variance by chunks of 100 rows
df = merged_df3.query("language=='English'")
chunk_size = int(len(df) * 0.1)  # 10% of the DataFrame's length
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i : i + chunk_size]
    print_3d_table_by_cols_name_fn(chunk, cutoffs=[0.1, 0.05])


# %%
# Check if it matters to send the whole conversation in at once or in pieces
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
# %%
# Test make_batch_requests lines up with previous results
merged_df3 = pd.read_pickle("data_dump/oai_mod/_temp_merged_df_fac53a6_base3_pt2.pkl")
for model, result_col in [
    ("text-embedding-3-small", "new_embedding_small"),
    ("text-embedding-3-large", "new_embedding"),
    ("text-embedding-ada-002", "new_embedding_ada002"),
    ("text-moderation-latest", "new_oai_mod"),
][3:]:
    print(model, result_col)
    df = merged_df3.head(50).copy()
    exp_result = (
        merged_df3[result_col].head(50).copy()
    )  # .apply(lambda x: pd.to_numeric(x, errors='coerce')))
    df.iloc[:25, df.columns.get_loc(result_col)] = None
    o = make_df_requests(df, "new_sent_convo", model, result_col=result_col)
    if result_col == "new_oai_mod":
        # f = lambda d: np.array(list(d["results"][0]["category_scores"].values()))
        unique_keys = o.apply(lambda d: tuple(d["results"][0]["category_scores"].keys())).unique()
        intersection = set.intersection(*[set(i) for i in unique_keys])
        categories = intersection
        f = lambda d: np.array(list(chat_max_by_cat(d, categories=categories).values()))
        o = o.apply(f)
        exp_result = exp_result.apply(f)
        atol = 0.03  # moderation's are a little wonky
    else:
        atol = 0.001
    assert o.combine(exp_result, lambda x, y: np.allclose(x, y, rtol=0, atol=atol)).all()
# %%
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
# filter to find why .nunique != len(set())
# assert merged_df["new_sent_convo"].apply(lambda d: d["content"]).nunique() == len(merged_df)

s, e = 0, len(merged_df)
model = (s + e) // 2
while s != model and model != e:
    df = merged_df["new_sent_convo"].iloc[s:model]
    if df.apply(lambda d: d["content"]).nunique() < len(df):
        e = model - 1
        print("good", s, model)
    else:
        s = model + 1
    model = (s + e) // 2
    print(s, model, e)
s, model = 0, 48
df = merged_df["new_sent_convo"][s:model]
print(
    df.apply(lambda d: d["content"]).nunique(), len(set(df.apply(lambda d: d["content"]))), len(df)
)
df.to_pickle("data_dump/data_df_where_content_nunique_doesnt_match_set_of_content")
# %%
# Longest prompts have bigger difference
data1 = merged_df["new_max_score"]
data2 = merged_df["default_max_score"]
i = 800
gt_800_tokens = merged_df["default_sent_convo"].apply(lambda d: num_tokens_from_messages([d])) > i
print(i)
fig, ax = plt.subplots(figsize=(7, 5))
_ks_hist_plot(data1[gt_800_tokens], data2[gt_800_tokens], ax=ax)
fig.suptitle(f"Original convo turn was >{i} tokens")
fig.subplots_adjust(top=0.86)
fig.savefig(
    f"plots/oai_mod/average_max_scores_yn_seperators_{git_hash()}_gt_800.png", facecolor="w"
)
# %%
# Longest prompts have bigger difference
data1 = merged_df["new_max_score"]
data2 = merged_df["default_max_score"]
i = 800
gt_800_tokens = merged_df["default_sent_convo"].apply(lambda d: num_tokens_from_messages([d])) > i
print(i)
fig, ax = plt.subplots(figsize=(7, 5))
_ks_hist_plot(data1[gt_800_tokens], data2[gt_800_tokens], ax=ax)
fig.suptitle(f"Original convo turn was >{i} tokens")
fig.subplots_adjust(top=0.86)
fig.savefig(
    f"plots/oai_mod/average_max_scores_yn_seperators_{git_hash()}_gt_800.png", facecolor="w"
)
# %%
# %% Random Scrap
p = np.arange(0, 1, 0.05)
print(
    analysis_mod_df["new_oai_mod"][some_mod]
    .apply(lambda d: chat_max_scores(d["results"]))
    .describe(percentiles=p),
    analysis_mod_df["new_oai_mod"][~some_mod]
    .apply(lambda d: chat_max_scores(d["results"]))
    .describe(percentiles=p),
)

# %% Random Scrap
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

# %%
long_convos = analysis_mod_df2.loc[
    analysis_mod_df2["sent_convo"].apply(num_tokens_from_messages) > 5000, "conversation_id"
]
# all convos over 10k got nans
old = (
    merged_df2.loc[
        merged_df2["conversation_id"].isin(long_convos), ["default_sent_convo", "conversation_id"]
    ]
    .groupby("conversation_id")["default_sent_convo"]
    .apply(lambda s: s.apply(num_tokens_from_messages).sum())
)
new = (
    merged_df2.loc[
        merged_df2["conversation_id"].isin(long_convos), ["new_sent_convo", "conversation_id"]
    ]
    .groupby("conversation_id")["new_sent_convo"]
    .apply(lambda s: s.apply(num_tokens_from_messages).sum())
)
# cut causes tokenization to be weird
(
    merged_df2["new_sent_convo"].apply(num_tokens_from_messages)
    - 2 * merged_df2["default_sent_convo"].apply(num_tokens_from_messages)
).value_counts()
(
    new
    - old * 2
    + 3
    * merged_df2[merged_df2["conversation_id"].isin(long_convos)].groupby("conversation_id").size()
)

# %%
# preembed_df4["new_embedding_small"] = batch_apply(
#    preembed_df4, "new_sent_convo", "text-embedding-3-small"
# )

bad_df = preembed_df4.iloc[bad_emd_ixs].copy()
print(bad_df.shape)
bad_df["new_embedding_small"] = batch_apply(
    bad_df,
    "new_sent_convo",
    "text-embedding-3-small",
)
bad_df2 = bad_df.head(2000)
print(bad_df2.shape)
bad_df2["new_oai_mod"] = batch_apply(
    bad_df2, "new_sent_convo", "text-moderation-latest", validate=True
)
print(
    "Total bad Token through 2 steps: ",
    bad_df2["default_sent_convo"].apply(num_tokens_from_messages).describe(),
)
sending_bad_df2 = bad_df2.query("new_any_flagged==False and new_max_score < 0.55")
print(
    "Total bad Token through filtering: ",
    sending_bad_df2["default_sent_convo"].apply(num_tokens_from_messages).describe(),
)

made_convo = make_finetune_convos1(bad_df2, convo_ids=sending_bad_df2.conversation_id.unique())
print(
    "Tokenings sending to finetune: ",
    made_convo["finetune_convo"]
    .apply(lambda d: num_tokens_from_messages(d["messages"]))
    .describe(),
)
# start_finetune(made_convo, "chat_bad_df2_ft1.json")
