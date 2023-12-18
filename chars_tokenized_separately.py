# %%
import requests
import pandas as pd
import json

# from pyarrow import parquet as pq
from concurrent.futures import ThreadPoolExecutor
import os

# HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]
HUGGING_FACE_API = "https://huggingface.co/api/datasets/lmsys/lmsys-chat-1m"

# Download first 2 train splits from:
# https://huggingface.co/datasets/lmsys/lmsys-chat-1m/tree/main/data
# https://huggingface.co/datasets/kjj0/4chanpol-openaimod/tree/main/data
ds_urls = {
    "lmsys-chat-1m": [
        "https://huggingface.co/datasets/lmsys/lmsys-chat-1m/resolve/main/data/train-00000-of-00006-4feeb3f83346a0e9.parquet",
        "https://huggingface.co/datasets/lmsys/lmsys-chat-1m/resolve/main/data/train-00001-of-00006-4030672591c2f478.parquet",
    ],
    "4chanpol-openaimod": [
        "https://huggingface.co/datasets/kjj0/4chanpol-openaimod/blob/main/data/train-00000-of-00048-6b6dfb39b513b835.parquet",
        "https://huggingface.co/datasets/kjj0/4chanpol-openaimod/blob/main/data/train-00001-of-00048-d041203d14b9a63b.parquet",
    ],
}


def download_file(url, local_filename, token):
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


with ThreadPoolExecutor(max_workers=4) as executor:
    files = list(
        executor.map(
            lambda ij: download_file(*ij, HUGGING_FACE_TOKEN),
            [
                (url, f"data_dump/{ds_name}/{url.split('/')[-1]}")
                for ds_name, urls in ds_urls.items()
                for url in urls
            ],
        )
    )

# files = [
#    "data_dump/lmsys-chat-1m/train-00000-of-00006-4feeb3f83346a0e9.parquet",
#    "data_dump/lmsys-chat-1m/train-00001-of-00006-4030672591c2f478.parquet",
#    "data_dump/4chanpol-openaimod/train-00001-of-00048-d041203d14b9a63b.parquet",
#    "data_dump/4chanpol-openaimod/train-00000-of-00048-6b6dfb39b513b835.parquet",
# ]

chat_df = pd.concat([pd.read_parquet(f) for f in files if "lmsys-chat-1m" in f])
completion_df = pd.concat([pd.read_parquet(f) for f in files if "4chanpol-openaimod" in f])


# %%
def _chat_is_flagged(openai_moderation):
    return any((r["flagged"] for r in openai_moderation))
    # flagging doesn't seem to line up wtih numbers?
    final_chat_df[categories][~final_chat_df["openai_moderation"].apply(_chat_is_flagged)].max(
        axis=0
    )
    # max where not flagged is 0.99s
    final_chat_df[categories][final_chat_df["openai_moderation"].apply(_chat_is_flagged)].min(
        axis=0
    )
    # min where flagged is all 0s


# chat_df["openai_moderation"].head().apply(_chat_is_flagged)

chat_df_flagged = chat_df[chat_df["openai_moderation"].apply(_chat_is_flagged)]
# Get top 100 for each harrasment category?
print(f"% flagged: {chat_df_flagged.size/chat_df.size*100:.1f}%, {len(chat_df_flagged)}")
# %%
categories = list(chat_df.loc[0, "openai_moderation"].values[0][0]["categories"].keys())


def _chat_max_by_cat(openai_moderation):
    return {c: max((r["category_scores"][c] for r in openai_moderation)) for c in categories}


chat_df = chat_df.join(
    pd.DataFrame(chat_df["openai_moderation"].apply(_chat_max_by_cat).values.tolist())
)

# categories with fewest first
categories = list(
    chat_df[categories][chat_df[categories] > 0.3].count().sort_values(ascending=True).keys()
)
# %%
N_PER_CATEGORY = 50
top_per_category = []
included_conversations = set()
df = chat_df.copy()
for category in categories:
    # Sort by category score and take the top 100
    unique_sorted_df = df.sort_values(by=[category], ascending=False).head(N_PER_CATEGORY)
    included_conversations.update(unique_sorted_df["conversation_id"])
    df = df[~df["conversation_id"].isin(included_conversations)]
    top_per_category.append(unique_sorted_df)

# Combine all the DataFrames
final_chat_df = pd.concat(top_per_category, ignore_index=True)
assert (
    sum(final_chat_df["openai_moderation"].apply(_chat_is_flagged))
    == len(categories) * N_PER_CATEGORY
), sum(final_chat_df["openai_moderation"].apply(_chat_is_flagged))
assert final_chat_df["conversation_id"].nunique() == len(final_chat_df)

print(
    final_chat_df[categories][final_chat_df[categories] > 0.3].count(),
    chat_df[categories][chat_df[categories] > 0.3].count(),
)


# %%
def is_flagged(openai_moderation):
    for record in json.loads(openai_moderation):
        if record.get("flagged", False):
            return True
    return False


def process_parquet_file(file_path):
    # Processing in chunks for memory efficiency
    chunk_size = 10000  # Adjust this based on your memory constraints
    for chunk in pd.read_parquet(file_path, chunksize=chunk_size):
        df_filtered = chunk[~chunk["openai_moderation"].apply(is_flagged)]
        yield df_filtered


def download_and_process(url):
    local_file = download_file(url, os.path.basename(url))
    for filtered_df in process_parquet_file(local_file):
        # Process or save each filtered chunk as needed
        print(filtered_df.head())


def ingest_data():
    urls = get_dataset_file_urls()
    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust number of threads
        executor.map(download_and_process, urls)


# Download first from this dataset

# filter for 'bad' things

# Record all seperator results

# %%
ord_use_between = [
    0,  # most unique
    8,  # delete char
    11,  # min matching  re.match('\s', chr(i)) in self_sync_ords
    # also min(works3) i.ie not changing ss1, ss2, h1
    190,  # max(works3)
    192,  # min matching regex.match('[^\r\n\p{L}\p{N}]?\p{L}+' in self_sync_ords
    # also min([i for n,i in num_sep_1_presuf if n == num_sep_1_presuf[1][0]])
    255,  # max([i for n,i in num_sep_1_presuf if n == num_sep_1_presuf[1][0]])
    1000,  # big and round, for randomness. not self_sync nor ascii
]
