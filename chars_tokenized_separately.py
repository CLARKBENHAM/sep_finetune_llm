# %%
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

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


# def download_file(url, local_filename, token):
#    headers = {"Authorization": f"Bearer {token}"}
#    with requests.get(url, headers=headers, stream=True) as r:
#        r.raise_for_status()
#        with open(local_filename, "wb") as f:
#            for chunk in r.iter_content(chunk_size=8192):
#                f.write(chunk)
#    return local_filename
#
#
# with ThreadPoolExecutor(max_workers=4) as executor:
#    files = list(
#        executor.map(
#            lambda ij: download_file(*ij, HUGGING_FACE_TOKEN),
#            [
#                (url, f"data_dump/{ds_name}/{url.split('/')[-1]}")
#                for ds_name, urls in ds_urls.items()
#                for url in urls
#            ],
#        )
#    )

files = [
    "data_dump/lmsys-chat-1m/train-00000-of-00006-4feeb3f83346a0e9.parquet",
    "data_dump/lmsys-chat-1m/train-00001-of-00006-4030672591c2f478.parquet",
    "data_dump/4chanpol-openaimod/train-00001-of-00048-d041203d14b9a63b.parquet",
    "data_dump/4chanpol-openaimod/train-00000-of-00048-6b6dfb39b513b835.parquet",
]
chat_df = pd.concat([pd.read_parquet(f) for f in files if "lmsys-chat-1m" in f], ignore_index=True)
completion_df = pd.concat(
    [pd.read_parquet(f) for f in files if "4chanpol-openaimod" in f], ignore_index=True
)


# %%
def _chat_is_flagged(openai_moderation):
    """If any message in convo is flagged"""
    return any((r["flagged"] for r in openai_moderation))


categories = list(chat_df.loc[0, "openai_moderation"][0]["categories"].keys())


def _chat_max_by_cat(openai_moderation):
    """Max score of any chat in convo"""
    return {c: max((r["category_scores"][c] for r in openai_moderation)) for c in categories}


def _chat_flagged_by_cat(openai_moderation):
    return {c: max((r["categories"][c] for r in openai_moderation)) for c in categories}


chat_df["any_flagged"] = chat_df["openai_moderation"].apply(_chat_is_flagged)
# Get top 100 for each harrasment category?
print(f"% flagged: {chat_df['any_flagged'].mean()*100:.1f}%, {chat_df['any_flagged'].sum()}")

chat_df = chat_df.join(pd.DataFrame(chat_df["openai_moderation"].apply(_chat_max_by_cat).tolist()))
# sort categories with fewest first
categories = list(
    chat_df[categories][chat_df[categories] > 0.3].count().sort_values(ascending=True).keys()
)

cat_flagged = pd.DataFrame(chat_df["openai_moderation"].apply(_chat_flagged_by_cat).values.tolist())
print(
    chat_df["any_flagged"].mean(),
    cat_flagged.mean(axis=0).sum(),
    "\n",
    cat_flagged.mean(axis=0).sort_values(),
)
d = chat_df["any_flagged"] != cat_flagged.apply(any, axis=1)
print(
    "Num flagged by category but not by total output",
    d.sum(),
    f"{d.mean()*100:.1f}%",
)
# %%
for c in categories:
    # basically random if use overall flagged y/n
    # plt.hist(chat_df[c][~chat_df["any_flagged"]], label="not flagged", alpha=0.3)
    # plt.hist(chat_df[c][chat_df["any_flagged"]], label="flagged", alpha=0.3)
    plt.hist(chat_df[c][~cat_flagged[c]], label="not flagged", alpha=0.3)
    plt.hist(chat_df[c][cat_flagged[c]], label="flagged", alpha=0.3)
    plt.title(c)
    plt.legend()
    plt.show()


# %%
# How can Hate have a high correlation and PCA component but not logistic regression coef?
def choose_columns(
    X, y, make_plots=False, n_ret_cols=None, min_pca_explained=0.95, n_pca_components=6
):
    # Make a heatmap of the covariance of the categories in chat_df
    corr = pd.concat([X, y], axis=1).corr()
    if make_plots:
        mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.figure(figsize=(10, 8))
        # TODO: doesn't label corr on each chart cell: fix versioning
        sns.heatmap(
            corr,
            annot=True,
            fmt=".1f",
            mask=mask,
            cmap="hot",
            xticklabels=corr.columns,
            yticklabels=corr.columns,
        )
        plt.show()
        # very little correlation:
        print_cut = 0.4
        print(f"Category pairs with correlation coefficient R >{print_cut}:")
        for i in range(len(corr.columns)):
            for j in range(i):
                if corr.iloc[i, j] > print_cut:
                    print(f"{corr.columns[i]} - {corr.columns[j]}: {corr.iloc[i, j]:.2f}")
        # if measuring aspects of the same thing or: violence - harassment/threatening: 0.51; harassment - hate: 0.61

    print("Column correlation with if flagged:")
    print(corr[y.name].sort_values())

    # Run a PCA on the corr matrix
    pca = PCA(n_components=n_pca_components)
    pca.fit(X)  # all 0-1 scale

    loadings = pd.DataFrame(
        pca.components_.T, columns=["PC%s" % _ for _ in range(n_pca_components)], index=X.columns
    )
    if make_plots:
        print(loadings)
        plt.plot(pca.explained_variance_ratio_)
        plt.ylabel("Explained Variance")
        plt.xlabel("Components")
        plt.show()
    if n_ret_cols is None:
        n_ret_cols = min(
            [
                i
                for i in range(1, n_pca_components)
                if sum(pca.explained_variance_ratio_[:i]) >= min_pca_explained
            ],
            default=n_pca_components,
        )
    loading_sums = np.sum(np.square(loadings), axis=1)
    print("Columns Explaining X(input) variance:\n", loading_sums.sort_values())
    pca_choosen_cols = np.argsort(loading_sums)[::-1][:n_ret_cols]
    pca_choosen_cols = list(pca_choosen_cols.index)
    return pca_choosen_cols

    # Cols most predictive of flagging by logistic reg
    # TODO: this doesn't make much sense but theoretically better captures what we want
    log = LogisticRegression(class_weight="balanced", fit_intercept=False)
    log.fit(X, y)

    # interpreting these right?
    logistic_most_contributing_ix = np.argsort(
        log.coef_[0],
    )[::-1]
    logisitic_most_contributing = np.array(categories)[logistic_most_contributing_ix[:n_ret_cols]]
    logistic_X = X[logisitic_most_contributing]
    # print(list(sorted(zip(log.coef_[0], log.feature_names_in_))))
    print(f"Log classifier score: {log.score(X, y):.3f}")
    print(f"        only {n_ret_cols} predictors {log.fit(logistic_X, y).score(logistic_X,y):.3f}")
    print(f"        only intercept {log.fit(np.ones((len(y),1)), y).score(np.ones((len(y),1)),y)}")
    print("Columns of the original dataset explaining the most variance:")
    print(logisitic_most_contributing)


X = chat_df[categories]
y = chat_df["any_flagged"]
test_columns = choose_columns(X, y)
X = chat_df[test_columns]
print(pd.concat([X, y], axis=0).corr())

# %%


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
# SCRAPE
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
