# %%
from google.cloud import aiplatform  # type: ignore
from google.protobuf.struct_pb2 import Value, Struct, ListValue
import asyncio
import os

## Issue is that it's not supported regardless of the API path.
# It's an issue on the google RPC end I belive
client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
llm_model = aiplatform.gapic.PredictionServiceAsyncClient(client_options=client_options)

def make_ip():
    i = Struct()
    t=Struct()
    t.fields['author'].MergeFrom(Value(string_value='user'))
    t.fields['content'].MergeFrom(Value(string_value='the prompt'))
    m = ListValue()
    m.values.append(Value(struct_value=t))
    i.fields["messages"].MergeFrom(Value(list_value=m))
    i.fields["anthropic_version"].MergeFrom(Value(string_value="vertex-2023-10-16"))
    i.fields["max_tokens"].MergeFrom(Value(string_value="256"))
    i.fields["stream"].MergeFrom(Value(string_value="false"))

    # i.fields["anthropic_version"].MergeFrom(Value(string_value="vertex-2023-10-16"))
    p = Struct()
    p.fields["maxDecodeSteps"].MergeFrom(Value(number_value=5))
    return i,p

def test(i,p, endpoint):
    print(asyncio.run(
                llm_model.predict(
                    endpoint = endpoint,
                    instances= [i],
                    parameters = p,
                )
            ))

i,p = make_ip()
# works sending to bison
test(i,p,endpoint = "projects/crypto-visitor-419221/locations/us-central1/publishers/google/models/chat-bison@002")
# fails sending to claude
test(i,p,endpoint="projects/crypto-visitor-419221/locations/us-central1/publishers/anthropic/models/claude-3-haiku@20240307",i,p)

# these endpoints work if called from CLI
os.popen("""
PROJECT_ID=crypto-visitor-419221
LOCATION=us-central1
MODEL=claude-3-haiku@20240307
cd ~/del
curl -X POST   -H "Authorization: Bearer $(gcloud auth print-access-token)"   -H "Content-Type: application/json; charset=utf-8"   -d @request.json "https://$LOCATION-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/$LOCATION/publishers/anthropic/models/$MODEL:streamRawPredict"
""").read()


#%%
def old_make_messages(messages):
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
    messages_list = old_make_messages(messages)
    struct_value.fields["messages"].MergeFrom(Value(list_value=messages_list))
    struct_value.fields["anthropic_version"].MergeFrom(Value(string_value="vertex-2023-10-16"))
    struct_value.fields["max_tokens"].MergeFrom(Value(number_value=max_tokens))
    struct_value.fields["stream"].MergeFrom(Value(bool_value=str(stream).lower() == "true"))
    instance = Value(struct_value=struct_value)
    return instance

def _prepare_request(message,max_output_tokens=10):
    parameters = {}

    max_output_tokens = max_output_tokens
    if max_output_tokens:
        parameters["maxDecodeSteps"] = max_output_tokens

    message_structs = []
    # add in old history
    message_structs.append(
        {
            "author": 'user',
            "content": message,
        }
    )

    instance = {"messages": message_structs}
    return instance, parameters
#i,p = _prepare_request('0. 0.8484239998505619',max_output_tokens=10)

#bad_real_messages = [{"role": "user", "content": [{"type": "text", "text": "What is in this image?"}]}]
#messages_list = make_message()
#real_messages = [{'author': 'user', 'content': '0. 0.8484239998505619'}]
i = Struct()
t=Struct()
t.fields['author'].MergeFrom(Value(string_value='user'))
t.fields['content'].MergeFrom(Value(string_value='the prompt'))
m = ListValue()
m.values.append(Value(struct_value=t))
i.fields["messages"].MergeFrom(Value(list_value=m))
# i.fields["anthropic_version"].MergeFrom(Value(string_value="vertex-2023-10-16"))
p = Struct()
p.fields["maxDecodeSteps"].MergeFrom(Value(number_value=5))
p.fields["anthropic_version"].MergeFrom(Value(string_value="vertex-2023-10-16"))
try:
    print(
        asyncio.run(
            llm_model.predict(
                # works: endpoint = "projects/crypto-visitor-419221/locations/us-central1/publishers/google/models/chat-bison@002" ,
                endpoint="projects/crypto-visitor-419221/locations/us-central1/publishers/anthropic/models/claude-3-haiku@20240307",
                instances= i, #{'messages': [{'author': 'user', 'content': '0. 0.8484239998505619'}]},
                parameters = p,
            )
        ),
    )
except Exception as e:
    print("not", type(e), e)


# sent on successful async
instances= {'messages': [{'author': 'user', 'content': '0. 0.8484239998505619'}]} #
parameters = {'maxDecodeSteps': 5}
self._model._endpoint._prediction_async_client
is google.cloud.aiplatform.utils.PredictionAsyncClientWithOverride

instances None <class 'vertexai.preview.language_models._PreviewChatModel'>
[{'messages': [{'author': 'user', 'content': '0. 0.8484239998505619'}]}]



b = _ChatSessionBase()
prediction_request = b._prepare_request(
            message=,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_sequences=stop_sequences,
            candidate_count=candidate_count,
            grounding_source=grounding_source,
        )
#What reformats from text input to correct output format
# (instance={'messages': [{'author': 'user', 'content': '0. 0.0001670967815873281'}]}, parameters={'maxDecodeSteps': 5})


# %%
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
                    endpoint="projects/crypto-visitor-419221/locations/us-central1/models/claude-3-haiku@20240307",
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
messages_list = old_make_messages(messages)
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
