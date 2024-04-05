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
    get_enc
)
from src.llm_requests import BatchRequests
from src.multi_model_tokenizers import AnthropicTokenizer, GemmaTokenizer, LlamaTokenizer

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

final_chat_df = pd.read_pickle("data_dump/final_chat_df_d6767b3.pkl")
final_chat_dfc = pd.read_pickle("data_dump/final_chat_dfc_f1978a7.pkl")


# Made a "Fill out the result frames for other models" in chars_tokenized_seperatly.py
gemini_results_framec = pd.read_pickle("data_dump/gemini/results_framec_c10ac43.pkl")
llama_results_framec = pd.read_pickle("data_dump/llama/results_framec_c10ac43.pkl")


class ResultsData:
    BR = BatchRequests(max_tokens=500)
    base_dir = "data_dump/combined_data"
    os.mkdir(base_dir)

    def __init__(
        self, og_chat_df_path: str, model: str, ord_vals: list[int], enc_name: str, context_window=4096,evals=list[str]
    ):
        self.model = model
        self.ord_vals = list(sorted([i for i in ord_vals if i is not None])) + [None]
        self.enc = get_enc(enc_name)
        self.completion_tokens = 500
        self.context_window = context_window
        self.og_chat_df_path = og_chat_df_path
        self.evals =
        #metadata_strs =
        #self.frame_paths = self.get_metadata_strs(is)
        #self.results_frame = None
        #self.results_path = [completion_{ms}.pkl" for ms in self.metadata_strs]
        #self.results = None

    def get_filenames(self, is_complete=False, evals=None):
        og_chat_file = os.path.basename(self.og_chat_df_path)
        completion_state = 'completion' if is_complete else 'frame'
        if evals is None:
            return [f"{ResultsData.base_dir}/{completion_state}-og_chat_file={og_chat_file}-model={self.model}-ord_val={o}-completion_tokens={self.completion_tokens}-context_window={self.context_window}" for o in self.ord_vals]
        return [f"{ResultsData.base_dir}/evals={e}-og_chat_file={og_chat_file}-model={self.model}-ord_val={o}-completion_tokens={self.completion_tokens}-context_window={self.context_window}" for o in self.ord_vals for e in evals]

    def frame_paths_to_make(self):
        new_frames = []
        for to_make, made in zip(self.get_filenames(is_complete=False),self.get_filenames(is_complete=True)):
            if not any((os.path.isfile(to_make), os.path.isfile(made))):
                new_frames += [to_make]
        return new_frames

    def parse_filename(filename):
        # Regular expression pattern to extract the components
        pattern = re.compile(r"/(?P<completion_state>\w+)-og_chat_file=(?P<og_chat_file>[\w\.]+)-model=(?P<model>[\w\-]+)-ord_val=(?P<ord_val>\w+)-completion_tokens=(?P<completion_tokens>\d+)-context_window=(?P<context_window>\d+)")
        match = pattern.search(filename)
        if match:
            data = match.groupdict()

            # Convert specific fields from strings to their correct types
            for int_field in ["completion_tokens", "context_window"]:
                data[int_field] = int(data[int_field]) if data[int_field].isdigit() else data[int_field]

            # Convert "ord_val" to either an integer, None, or leave as a string
            if data["ord_val"] == "None":
                data["ord_val"] = None
            elif data["ord_val"].isdigit():
                data["ord_val"] = int(data["ord_val"])

            return data
        else:
            assert False, filename

    def _new_result_frames(self):
        new_paths = []
        for frame_path in self.frame_paths_to_make():
            data = self.parse_filename(frame_path)
            og_chat_df = pd.read_pickle(self.og_chat_df_path)

            # make results frame the same way, but also include a column for

            new_paths += [frame_path]
        return new_paths

    async def _fill_out_frame(self, frame_path):
        # pd.read results_path, only make below if not possible. add to self.results object
        data = self.parse_filename(frame_path)
        # makes requests using single BR object so the rate limiting holds
        results = await ResultsData.BR.batch_router_requests(list_of_req)

        # Make the list of requirements like I did previously using code from fill_out_results
        # this time use the BR object to make the moderation results.

    def get_results(self):
        asyncio.run(asyncio.gather(*[self._fill_out_frame(path) for path in self._new_result_frames()]))
        return pd.concat([pd.read_pickle(f) for f in self.get_filenames(is_complete=True)])

    async def add_evals(self, evals):

        return pd.concat([pd.read_pickle(f) for f in self.get_filenames(evals=evals)])

class AnalysisData:
    def __init__(self, results_data: ResultsData, evals: list[str]):
        self.results_data = results_data
        evals = list(sorted(evals))
        self.evals = evals
        self.eval_paths = [f"{ResultsData.base_dir}/{self.results_data.metadata_str}_eval={e}.pkl" for e in evals]
        self.analysis_path = f"{ResultsData.base_dir}/{self.results_data.metadata_str}_{"".join(evals)}analysis.pkl"
        self.analysis = None

class Experiment():
    def __init__()
        self.results_datas =

# Now how can I combine all these experiments into 1 final frame and run everything at once?
experiment1 = ResultsData(og_chat_df_path="data_dump/final_chat_df_d6767b3.pkl", model=["gpt-3.5" ], ord_vals=[8, 192], enc_name='openai')
experiment1 = ResultsData(og_chat_df_path="data_dump/final_chat_df_d6767b3.pkl", model=["gpt-4"], ord_vals=[8, 192], enc_name='openai')
experiment1 = ResultsData(og_chat_df_path="data_dump/final_chat_df_d6767b3.pkl", model=['claude-3-haiku-20240307' ], ord_vals=[8, 192], enc_name='anthropic')
experiment1 = ResultsData(og_chat_df_path="data_dump/final_chat_df_d6767b3.pkl", model=['gemini-pro'], ord_vals=[8, 192], enc_name='gemma')




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


def fill_out_results(df_frame, n_loops=1, req_client=oa_client):
    with ThreadPoolExecutor(max_workers=15) as executor:
        results_df = copy.deepcopy(df_frame)
        missing_ix = results_df["new_oai_mod"].isna()
        while sum(missing_ix) and n_loops > 0:
            print(f"adding {sum(missing_ix)}")
            m_completion, m_oai_mod = list(
                zip(
                    *executor.map(
                        lambda mcsep: get_chat_completion(
                            mcsep[0], mcsep[1], sep=mcsep[2], req_client=req_client
                        ),
                        zip(
                            results_df["new_model"][missing_ix],
                            results_df["sent_convo"][missing_ix],
                            results_df["manipulation"][missing_ix].apply(lambda i: i["sep"]),
                        ),
                    )
                )
            )
            print("num new: ", sum([i is not None for i in m_oai_mod]))
            results_df.loc[missing_ix, "new_completion"] = m_completion
            m_oai_mod2 = [o[0] if isinstance(o, list) else o for o in m_oai_mod]
            results_df.loc[missing_ix, "new_oai_mod"] = m_oai_mod2
            missing_ix = results_df["new_oai_mod"].isna()
            n_loops -= 1

    results_df["new_oai_mod"] = results_df["new_oai_mod"].apply(
        lambda o: o if isinstance(o, list) or o is None or o is np.nan else [o]
    )
    return results_df


# %%
an_results_framec = pd.read_pickle(f"data_dump/an_mod/results_framec_{git_hash()}.pkl")


# %%
