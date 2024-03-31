# Things for CAA activation steering; poster
import pandas as pd
import json

from src.utils import between_tokens, end_of_convo, chat_to_str, git_hash


final_chat_df = pd.read_pickle("data_dump/final_chat_df_d6767b3.pkl")
final_chat_dfc = pd.read_pickle("data_dump/final_chat_dfc_f1978a7.pkl")
analysis_df = pd.read_pickle("data_dump/analysis_df_new_model_02_13_f17eae7.pkl")
analysis_dfc = pd.read_pickle("data_dump/analysis_dfc_02_02_f1978a7.pkl")


def get_top(final_chat_dfc, analysis_dfc):
    sim_sent = analysis_dfc.query('new_model=="gpt-4-0613"')
    sim_sent = sim_sent[sim_sent["manipulation"].apply(lambda d: ord(d["sep"]) == 192)]
    sim_sent.columns
    difference = sim_sent["new_max_scores"] - sim_sent["gpt40613_max_scores"]
    indices_of_largest = difference.nlargest(50).index
    indices_of_largest = list(indices_of_largest)
    caa_final_chat_df = final_chat_dfc.loc[indices_of_largest]
    return caa_final_chat_df


caa_final_chat_df = get_top(final_chat_dfc, analysis_dfc)
caa_final_chat_df_test = get_top(final_chat_df, analysis_df)


def _make_caa_results_json(
    ord_vals=[192], model="gpt-4-0613", enc=None, file=None, _chat_dfc=caa_final_chat_df
):
    mx_toks = 1500  # floor((4096-100)/1.5)

    # make_new_convo=lambda r: prepend_prompt(
    #    make_prompt13,
    #    r["manipulation"]["sep"],
    #    r["sent_convo"],
    #    role="system",
    # )
    new_dfs = []
    for ord_val in ord_vals:
        _r_df = pd.DataFrame(index=_chat_dfc.index).reset_index()

        _r_df["question"] = ""
        _r_df["model"] = model
        _r_df["encoding"] = enc
        sep = chr(ord_val)
        _r_df["manipulation"] = [{"kind": "between", "sep": sep, "has_system_prompt": False}] * len(
            _r_df
        )

        # print(_chat_dfc["conversation"].apply(lambda l: sum(map(len,l))).describe())
        short_text = (
            _chat_dfc["conversation"]
            .apply(
                lambda convo: chat_to_str(
                    end_of_convo(
                        convo, max_tokens=mx_toks, enc="llama70b", strip_first_assistant=False
                    )
                )
            )
            .copy()
        )
        # print(short_text.apply(len).describe())

        _r_df["answer_not_matching_behavior"] = short_text
        # sep token is matching behavior
        # _r_df["sent_convo"] =  # if using _r_df.apply(make_new_convo, axis=1)

        _r_df["answer_matching_behavior"] = short_text.apply(
            lambda s: between_tokens(s, sep, enc=enc)
        )

        # del _r_df["sent_convo"]

        new_dfs += [_r_df]

    data_as_dict = pd.concat(new_dfs).to_dict(orient="records")
    with open(file, "w", encoding="utf-8") as file:
        json.dump(data_as_dict, file, ensure_ascii=False, indent=4)
    return data_as_dict


oa_caa_json_hack = _make_caa_results_json(
    model="gpt-4-0613", enc="openai", file=f"data_dump/oa_caa_json_hack_{git_hash()}.json"
)
gemini_caa_json_hack = _make_caa_results_json(
    model="gemini-pro",
    enc="gemma7b",
    file=f"data_dump/gemini/gemma7b_caa_json_hack_{git_hash()}.json",
)
llama_caa_json_hack = _make_caa_results_json(
    model="llama-70b",
    enc="llama2",
    file=f"data_dump/llama/llama70b_caa_json_hack_{git_hash()}.json",
)
llama_caa_json_hack_test = _make_caa_results_json(
    model="llama-70b",
    enc="llama2",
    file=f"data_dump/llama/llama70b_caa_json_hack_test_{git_hash()}.json",
    _chat_dfc=caa_final_chat_df_test,
)
