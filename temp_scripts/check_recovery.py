# Temp code to record that everything lines up
# would need run in ./chars_tokenized_seperately.py
# %%
final_chat_df = pd.read_pickle("data_dump/final_chat_df_d6767b3.pkl")
final_chat_dfb = pd.read_pickle("data_dump/final_chat_dfb_2e513e8.pkl")
final_chat_df2 = pd.read_pickle("data_dump/final_chat_df2_d6767b3.pkl")
final_chat_df3 = pd.read_pickle("data_dump/final_chat_df3_d6767b3.pkl")

# only for results_df are emptys are loaded as nans not ''
results_df = pd.read_pickle("data_dump/results_df_01_24_b511c0f.pkl")
# results_df["new_completion"][results_df["new_completion"].isna()] = ""
# results_df.to_pickle("data_dump/results_df_01_24_b511c0f.pkl")

results_dfb = pd.read_pickle("data_dump/results_dfb_01_30_2e513e8.pkl")
# results_df2 has 2 missing values, not sure oai wouldn't create completions for those
results_df2 = pd.read_pickle("data_dump/_bad_results2_01_25_34d63d4.pkl")
results_df3 = pd.read_pickle("data_dump/results_df3_01_26_7486c8c.pkl")

analysis_df = make_analysis_df(results_df, final_chat_df)
analysis_dfb = make_analysis_df(results_dfb, final_chat_dfb)
# analysis_df.to_pickle(f"data_dump/analysis_df_01_30_3227533.pkl") # replacing nans
analysis_df2 = make_analysis_df(results_df2[~results_df2["new_oai_mod"].isna()], final_chat_df2)
analysis_df3 = make_analysis_df(results_df3, final_chat_df3)

# fixed ix
# exp_analysis_df = pd.read_pickle(f"data_dump/analysis_df_01_30_3227533.pkl")
# exp_analysis_df.set_index("Unnamed: 0", inplace=True, drop=True)
# if set(exp_analysis_df.columns) != set(analysis_df.columns):
#    raise ValueError("The columns of 'exp_analysis_df' and 'analysis_df' are not the same.")
# else:
#    exp_analysis_df = exp_analysis_df.reindex(columns=analysis_df.columns)
# exp_analysis_df.to_pickle(f"data_dump/analysis_df_01_30_3227533.pkl")

exp_analysis_df = pd.read_pickle(f"data_dump/analysis_df_01_30_3227533.pkl")
exp_analysis_dfb = pd.read_pickle(f"data_dump/analysis_dfb_01_30_2e513e8.pkl")
exp_analysis_df2 = pd.read_pickle(f"data_dump/analysis_df2_01_25_34d63d4.pkl")
exp_analysis_df3 = pd.read_pickle(f"data_dump/analysis_df3_01_26_7486c8c.pkl")
# %%
print(
    "results",
    results_df.isna().sum().sort_values().tail(),
    results_dfb.isna().sum().sort_values().tail(),
    results_df2.isna().sum().sort_values().tail(),
    results_df3.isna().sum().sort_values().tail(),
)

print(
    "analysis",
    analysis_df.isna().sum().sort_values().tail(),
    analysis_dfb.isna().sum().sort_values().tail(),
    analysis_df2.isna().sum().sort_values().tail(),
    analysis_df3.isna().sum().sort_values().tail(),
)
print(
    "exp_analysis",
    exp_analysis_df.isna().sum().sort_values().tail(),
    exp_analysis_dfb.isna().sum().sort_values().tail(),
    exp_analysis_df2.isna().sum().sort_values().tail(),
    exp_analysis_df3.isna().sum().sort_values().tail(),
)
# %%
# all the diffs are from rounding or nans
# but analysis_df got overwritten so identical now
for c in analysis_df.columns:
    print(c)
    try:
        if np.sum(exp_analysis_df[c] != analysis_df[c]) > 0:
            # ix1, ix2 = ~exp_analysis_df[c].isna(), ~analysis_df[c].isna()
            print(c, len(exp_analysis_df[c].compare(exp_analysis_df[c])))
    except Exception as e:
        print("FAILED", c, e)
print(
    "diff only nans",
    (exp_analysis_df["new_hate"] - analysis_df["new_hate"]).agg(["max", "min", "mean", "std"]),
)
c = "new_completion"
f = lambda i: len(i) if isinstance(i, str) else -1
np.sum(exp_analysis_df[c].apply(f).values == analysis_df[c].apply(f).values)
print(exp_analysis_df.isna().sum().sort_values(), analysis_df.isna().sum().sort_values())
# diffs only nans
e = pd.read_pickle(f"~/del/analysis_df_01_30_3227533.pkl")
e.set_index("Unnamed: 0", inplace=True, drop=True)
cols = ["new_completion", "gpt40613_completion", "gpt41106preview_completion"]
for c in cols:
    d = e[c].compare(analysis_df[c])
    assert d["self"].isna().all()
    assert (d["other"].apply(f) <= 0).all()
# %%
# for c in analysis_dfb.columns:
for c in analysis_df.columns:
    print(c)
    try:
        # print(np.sum(exp_analysis_dfb[c] == analysis_dfb[c]))
        print(np.sum(exp_analysis_df2[c] == analysis_df2[c]))
        print(np.sum(exp_analysis_df3[c] == analysis_df3[c]))
    except:
        print("FAILED", c)
# %%
# analysis_df2 only diff from expected because of the nans
for c in analysis_df2.columns:
    try:
        if np.sum(exp_analysis_df2[c] != analysis_df2[c]) > 0:
            ix1, ix2 = ~exp_analysis_df2[c].isna(), ~analysis_df2[c].isna()
            print(c, np.sum(exp_analysis_df2[c][ix1] != analysis_df2[c][ix2]))
    except:
        print("FAILED", c)


# %% Old way of checks

# WARN: results_df was slightly different sent_convo from expected
exp_final_chat_df = recover_csv("data_dump/preprocessing_chat_df_250_34d63d4.csv")
exp_results_df = recover_csv("data_dump/results_01_24_beb23c3.csv", index_col=0)
exp_analysis_df = recover_csv("data_dump/analysis_df_01_24_beb23c3.csv", index_col=0)

final_chat_dfb = pd.read_pickle("data_dump/final_chat_dfb_2e513e8.pkl")
results_dfb = pd.read_pickle("data_dump/resultsb_01_30_2e513e8.pkl")
analysis_dfb = pd.read_pickle("data_dump/analysis_dfb_01_30_2e513e8.pkl")

# Not really going to use results2 data, didn't test on gpt411 but 8 wo were above 0.3
# and 2 rows got type errors with no manipulation for both models
# these 2 na rows in oai_mod and completion otherwise break the analysis_df
exp_final_chat_df2 = pd.read_pickle("data_dump/final_chat_df2_250_34d63d4.pkl")
exp_results_df2 = pd.read_pickle("data_dump/results2_01_25_34d63d4.pkl")
exp_analysis_df2 = pd.read_pickle("data_dump/analysis_df2_01_25_34d63d4.pkl")

exp_final_chat_df3 = pd.read_pickle("data_dump/final_chat_df3_d6767b3.pkl")
exp_results_df3 = pd.read_pickle("data_dump/results3_01_26_7486c8c.pkl")
exp_analysis_df3 = pd.read_pickle("data_dump/analysis_df3_01_26_7486c8c.pkl")

assert final_chat_df[exp_final_chat_df.columns].equals(exp_final_chat_df)
print(
    exp_final_chat_df.equals(final_chat_df),  # false from column order
    # exp_results_df.equals(results_df),
    # exp_analysis_df.equals(analysis_df),
    exp_final_chat_df2.equals(final_chat_df2),
    # exp_results_df2.equals(results_df2), # results different from sent_convo, seems encoding changed(?!)
    # exp_analysis_df2.equals(analyssi_df2),
)

# results slightly different, mostly for None's (!)
# print(results_frame2[results_df2.columns].compare(results_df2))
print(
    np.sum(exp_results_df["sent_convo"].apply(str) == _mrf(final_chat_df)["sent_convo"].apply(str))
)  # 1834
print(
    np.sum(
        exp_results_df2["sent_convo"].apply(str) == _mrf(final_chat_df2)["sent_convo"].apply(str)
    )
)  # 1991
print(
    Counter(
        results_frame2[
            results_frame2["sent_convo"].apply(str) != exp_results_df2["sent_convo"].apply(str)
        ]["manipulation"].map(lambda d: d["sep"])
    ),
    Counter(
        results_frame[
            results_frame["sent_convo"].apply(str) != exp_results_df["sent_convo"].apply(str)
        ]["manipulation"].map(lambda d: d["sep"])
    ),
)
print(
    exp_final_chat_df2["conversation"].apply(str).nunique(),
    exp_results_df2["sent_convo"].apply(str).nunique(),
    exp_final_chat_df["conversation"].apply(str).nunique(),
    exp_results_df["sent_convo"].apply(str).nunique(),
)
# final_chat_df had 3 dups, 247*8=1976 so that checks
# a = recover_csv("data_dump/preprocessing_chat_df_250.csv")
# # weirdness around utf-8, encoding/decoding with replace
# df = final_chat_df[a.columns].compare(a)
# def safe_encode(x):
#    if isinstance(x, str) or hasattr(x, "encode"):
#        return x.encode("utf-8", "ignore").decode("utf-8", "ignore")
#    return x
# df = final_chat_df[a.columns].compare(a)
# for column in df.columns:
#    df[column] = df[column].apply(safe_encode)
# print(df[~df["conversation"]["self"].isna()]["conversation"])
# a = recover_csv("data_dump/results_01_23.csv", index_col=0)
# Not quite right
# a2 = recover_csv("data_dump/results2_01_25_34d63d4.csv", index_col=0)
# c = [c for c in a2.columns if c != "sent_convo"]
# results_df2[c].compare(a2[c])

# results_df = recover_csv("data_dump/results_01_23.csv", index_col=0)
# analysis_df2 = pd.read_pickle("data_dump/analysis_df2_01_25_34d63d4.pkl")
