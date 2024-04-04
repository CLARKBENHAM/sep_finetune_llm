# %%
from google.cloud import aiplatform  # type: ignore
from google.protobuf.struct_pb2 import Value, Struct, ListValue
import asyncio


def make_message(messages):
    messages_list = ListValue()  # Create the messages ListValue
    for message in messages:
        message_struct = Struct()
        message_struct.fields["role"].MergeFrom(Value(string_value=message["role"]))

        content_list = ListValue()
        for content_item in message["content"]:
            content_value = Struct()
            content_value.fields["type"].MergeFrom(Value(string_value=content_item["type"]))
            content_value.fields["text"].MergeFrom(Value(string_value=content_item["text"]))
            content_list.values.append(
                Value(struct_value=content_value)
            )  # Append Struct wrapped in a Value

        message_struct.fields["content"].MergeFrom(
            Value(list_value=content_list)
        )  # Merge the ListValue
        messages_list.values.append(
            Value(struct_value=message_struct)
        )  # Append Struct wrapped in a Value
    return messages_list


def create_instance(messages, max_tokens, stream=True):
    """Creates a single instance with the required structure."""
    struct_value = Struct()
    messages_list = make_message(messages)
    struct_value.fields["messages"].MergeFrom(Value(list_value=messages_list))
    struct_value.fields["anthropic_version"].MergeFrom(Value(string_value="vertex-2023-10-16"))
    struct_value.fields["max_tokens"].MergeFrom(Value(number_value=max_tokens))
    struct_value.fields["stream"].MergeFrom(Value(bool_value=str(stream).lower() == "true"))
    instance = Value(struct_value=struct_value)
    return instance


messages = [{"role": "user", "content": [{"type": "text", "text": "What is in this image?"}]}]
max_tokens = 256

instances = [create_instance(messages, max_tokens)]

client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
llm_model = aiplatform.gapic.PredictionServiceAsyncClient(client_options=client_options)
# using either endpoint gives "400 invalid argument"
# endpoint='projects/crypto-visitor-419221/locations/us-central1/models/claude-3-haiku@20240307', # used in model
# endpoint="projects/crypto-visitor-419221/locations/us-central1/models/claude-3-haiku@20240307"
asyncio.run(
    llm_model.predict(
        # endpoint='projects/crypto-visitor-419221/locations/us-central1/models/claude-3-haiku@20240307', # used in model
        endpoint=(
            "projects/crypto-visitor-419221/locations/us-central1/models/claude-3-haiku@20240307"
        ),
        instances=instances,
    )
)
# %%
# OG vertex
""" [struct_value {
  fields {
    key: "vertex_publisher"
    value {
      string_value: "anthropic"
    }
  }
  fields {
    key: "prompt"
    value {
      string_value: "0. 0.4171215570602269"
    }
  }
  fields {
    key: "max_tokens"
    value {
      number_value: 5
    }
  }
  fields {
    key: "max_retries"
    value {
      number_value: 0
    }
  }
}
]
"""
from copy import deepcopy
from itertools import combinations, chain


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


for t in powerset(["vertex_publisher", "prompt", "max_tokens", "max_retries"]):
    i = deepcopy(instances)
    print(t)
    for f in t:
        del i[0].struct_value.fields[f]

    try:
        print(
            t,
            asyncio.run(
                llm_model.predict(
                    endpoint="projects/crypto-visitor-419221/locations/us-central1/endpoint/claude-3-haiku@20240307",
                    instances=i,
                )
            ),
        )
    except Exception as e:
        print("not", t, type(e), e)


i = deepcopy(instances)

del i[0].struct_value.fields["prompt"]
del i[0].struct_value.fields["max_retries"]
type(llm_model)

# %%

i = Struct()
messages_list = make_message(messages)
i.fields["messages"].MergeFrom(Value(list_value=messages_list))
try:
    print(
        t,
        asyncio.run(
            llm_model.predict(
                endpoint="projects/crypto-visitor-419221/locations/us-central1/publishers/anthropic/models/claude-3-haiku@20240307",
                instances=i,
                parameters={"maxDecodeSteps": 5},
                # parameters={
                #    "anthropic_version": "vertex-2023-10-16",
                #    "messages": [{"role": "user", "content": "Send me a recipe for banana bread."}],
                #    "max_tokens": 100,
                #    "stream": True,
                # },
            )
        ),
    )
except Exception as e:
    print("not", t, type(e), e)
