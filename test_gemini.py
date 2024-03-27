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
    get_mod,
    chat_to_str,
    num_tokens_from_messages,
    num_tokens_from_string,
    MX_TOKENS,
    end_of_convo,
    take_last_tokens,
    git_hash,
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

HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]
# HUGGING_FACE_API = "https://huggingface.co/api/datasets/lmsys/lmsys-chat-1m"
pd.set_option("display.max_colwidth", 1000)

# %%




