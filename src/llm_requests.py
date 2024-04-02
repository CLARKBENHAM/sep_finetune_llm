# %%
import importlib

# importlib.reload(litellm)
from litellm import completion, batch_completion, Router
import os
import asyncio
import time
import math
from collections import deque
import random
import nest_asyncio

nest_asyncio.apply()

# Already set all API's keys in bashrc:
for k in "OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY".split(", "):
    assert k and len(os.environ[k])

# Will keep below limits for each model
# but then can't batch requests with router
# how to limit size of completions?


model_list = [  # list of model deployments
    {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {  # params for litellm completion/embedding call
            "model": "gpt-3.5-turbo",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "max_tokens": 5,
        },
        # "tpm": 2,
        "rpm": 20,
        "tpm": 500,
        # "rpm": 15000,
    },
    {
        "model_name": "gpt-4",
        "litellm_params": {
            "model": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "max_tokens": 5,
        },
        # "tpm": 2,
        "rpm": 4,
        "tpm": 2000,
        # "rpm": 15000,
    },
]

router = Router(
    model_list=model_list,
    routing_strategy="usage-based-routing",
    debug_level="DEBUG",
    #    num_retries=3,
    #    retry_after=5,
)
router2 = Router(
    model_list=model_list,
    routing_strategy="usage-based-routing",
    debug_level="DEBUG",
)


# Async doesn't error
def run_does_error(i):
    print(time.time())
    return router.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"{i}. Hey, how's it going? {random.random()}"}],
    )


# Only errors if re-use router1
# async def arun_doesnt_error(i):
#    response = await router.acompletion(
#        model="gpt-3.5-turbo",
#        messages=[{"role": "user", "content": f"{i}. Hey, how's it going? {random.random()}"}],
#    )
#    return response


async def arun_doesnt_error(i):
    response = await router2.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"{i}. Hey, how's it going? {random.random()}"}],
    )
    return response


async def does_error(data, sync=True):
    try:
        if sync:
            # can't start multiple requests at once if do it sync
            # results = [(print(i), run_does_error(i)) for i in data]
            results = [run_does_error(i) for i in data]
        else:
            tasks = [arun_doesnt_error(i) for i in data]
            results = await asyncio.gather(*tasks)
        print(results)
        print(sum([1 for i in results if i is not None]))
        return False
    except Exception as e:
        print(e)
        return True


# asyncio.run(does_error(range(666), sync=True)) # You can't start multiple requests at once

# If run below syncronously then rate limit errors occur
# assert asyncio.run(does_error(range(6), sync=True))
# but no rate limit errors if start all request async
# assert asyncio.run(does_error(range(300), sync=False))
# But now that asyncio ends, router2 will update rate limits and realizes there's not an availible gpt-3.5-turbo
print(router2.get_available_deployment("gpt-3.5-turbo"))
data = range(30000)
tasks = [arun_doesnt_error(i) for i in data]
results = asyncio.run(asyncio.gather(*tasks))
print(len(results), len([i for i in results if i is not None]))

# %%


async def arun_doesnt_error(router, i):
    response = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"{i}. Hey, how's it going? {random.random()}"}],
    )
    return response


async def test_run_doesnt_error(router, data):
    tasks = [arun_doesnt_error(router, i) for i in data]
    results = await asyncio.gather(*tasks)
    return results


results = asyncio.run(test_run_doesnt_error(router, range(20)))
print(sum([i for i in results if i is None]), results)

# %%
ne = 0

BURST_START_INTERVAL = 0.1  # don't start eg. 10k requests all at once, make it less bursty
RPM_TO_RPBURST = 2.5  # limit over min * this to be limit over REQUEST_START_INTERVAL


# So run doesn't rate limit within router, but it will within main.
async def run(router, rate_limit_queue, request_start_time_queue, model, messages):
    global ne
    i = 0
    while i < 5:
        max_rpm = max(
            (m["litellm_params"].get("rpm", 0) for m in router.healthy_deployments), default=100
        )
        MAX_STARTS_PER_INTERVAL = max_rpm * RPM_TO_RPBURST * 60 / BURST_START_INTERVAL

        now = time.time()
        while request_start_time_queue and request_start_time_queue[0] <= now:
            request_start_time_queue.popleft()
        if len(request_start_time_queue) >= MAX_STARTS_PER_INTERVAL:
            next_available_start_time = request_start_time_queue[0]
            await asyncio.sleep(next_available_start_time - now)
            continue
        else:
            try:
                rate_limit_queue.append(now + 60)
                request_start_time_queue.append(now + BURST_START_INTERVAL)
                response = await router.acompletion(model, messages)
                # response = router.completion(model, messages)
                print("result at", time.time(), now)
                return response
            except Exception as e:
                if isinstance(e, ValueError):
                    ne += 1
                    print(ne, e)
                    # retrying
                    if rate_limit_queue:
                        next_available_start_time = rate_limit_queue.popleft()
                        if next_available_start_time > now:
                            print("sleeping till", next_available_start_time)
                            await asyncio.sleep(next_available_start_time - now)
                    i += 1
                    pass
                else:
                    # Unexpected error
                    print(f"Error: {e} {type(e)}")
                    return None
    return None


import random


async def main(data, router):
    rate_limit_queue = deque()
    request_start_time_queue = deque()

    tasks = [
        run(
            router,
            rate_limit_queue,
            request_start_time_queue,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"{i}. Which model number are you? {random.random()}"}
            ],
        )
        for i in data
    ]
    results = await asyncio.gather(*tasks)
    return results


# Why do all 20 requests fail now?
# results = asyncio.run(main(range(10), router))

# Run the main function
results = [asyncio.run(run(i)) for i in range(15000)]
print(ne, sum([1 for i in results if i is not None]))
print([r.model for r in results])
# router.get_available_deployment("gpt-3.5-turbo")
# %%

# litellm.set_verbose = True
s = "Who are you?"
for m in ["gemini/gemini-pro", "claude-3-haiku-20240307", "gpt-4-0125-preview"]:
    messages = [{"role": "user", "content": s}]
    print(m, completion(m, messages=messages))
    print(
        m,
        batch_completion(
            m, messages=[[{"role": "user", "content": f"{i}. {s}"}] for i in range(3)]
        ),
    )


# Handle it to api level

#
# %%
import tenacity

import google.generativeai as genai
import os

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import openai

from anthropic import Anthropic

### Gemini Code
genai.configure(api_key=os.environ["GOOGLE_AI_STUDIO_API_KEY"])
model = genai.GenerativeModel("gemini-pro")

# Generative
response = model.generate_content("Please summarise this document: ...")
# Chat
chat = model.start_chat()


def get_chat_response(chat: ChatSession, prompt: str) -> str:
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)


chat = "Hello."
print(get_chat_response(chat, prompt))


# Anthropic Code


# OpenAI Code
oa_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

response = oa_client.chat.completions.create(
    model=model,
    messages=messages,
    **kwargs,
)


from abc import ABC, abstractmethod


class CompletionStrategy(ABC):
    @abstractmethod
    def chat_completion(self, chat):
        pass

    @abstractmethod
    def string_completion(self, input_string):
        pass


class OpenAIStrategy(CompletionStrategy):
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        # Assume the SDK is initialized here

    def chat_completion(self, chat):
        # Placeholder for the actual OpenAI SDK call
        return f"OpenAI chat completion for: {chat}"

    def string_completion(self, input_string):
        # Placeholder for the actual OpenAI SDK call
        return f"OpenAI string completion for: {input_string}"


class AnthropicStrategy(CompletionStrategy):
    def __init__(self, api_key):
        self.client = Anthropic(api_key)
        # Assume the SDK is initialized here

    def chat_completion(self, chat):
        # Placeholder for the actual Anthropic SDK call
        message = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": chat}],
        )
        return f"Anthropic chat completion for: {message.content}"

    def string_completion(self, input_string):
        # Placeholder for the actual Anthropic SDK call
        client = Anthropic(api_key=self.api_key)
        message = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": input_string}],
        )
        return f"Anthropic string completion for: {message.content}"


class GeminiStrategy(CompletionStrategy):
    def __init__(self, api_key):
        self.client = Gemmin(api_key)
        # Assume the SDK is initialized here

    def chat_completion(self, chat):
        # Placeholder for the actual Gemini SDK call
        return f"Gemini chat completion for: {chat}"

    def string_completion(self, input_string):
        # Placeholder for the actual Gemini SDK call
        return f"Gemini string completion for: {input_string}"


# Take final_chat_df, make results frame:
# For all model/encoding pairs. openai
# For Model/another encoder pairs, eg anthropic on llama tokenizer
# 	for anthropic Or for Anthropic need another approach, breaking it up into smaller sentences

# take results frame, make full results
# need fn that takes model name, returns "strategy and rate limits"

# munge results frame into analysis df

# easy comparisons, but now the column names keep changing


def get_chat_completion(model, s, sep=None, req_client=oa_client, **kwargs):
    # always use OAI 'client' for mod
    if isinstance(s, str):
        messages = [
            {
                "role": "user",
                "content": s,  # f"Continue this story with {sep}:```{s}```", # also makes words 'worse'
            }
        ]
    else:
        messages = s
        if isinstance(messages, np.ndarray):
            messages = list(s)  # if np arr then screws up
    for i in range(4):
        try:
            kwargs["stop"] = getattr(
                kwargs, "stop", ["Sorry, ", "I'm sorry", "I apologize", "I'm really sorry"]
            )
            kwargs["max_tokens"] = getattr(kwargs, "max_tokens", 500)

            response = req_client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
        except Exception as e:
            try:
                print(e.status_code // 100 == 4, e)
                if e.status_code // 100 == 4:
                    return None, None
            except Exception as e:
                print(e)
                # print(model, s, sep, client, kwargs)
                return None, None
            time.sleep(1.5**i)
        else:
            out = response.choices[0].message.content
            if sep is not None:
                out = out.replace(sep, "")
            for i in range(4):
                try:
                    mod = oa_client.moderations.create(input=out, model="text-moderation-stable")
                except:
                    time.sleep(1.5**i)  # haven't every seen this rate limit
                else:
                    return out, mod.model_dump()["results"]
    return None, None


# %%
i = 19
router.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": f"{i}. Hey, how's it going? {random.random()}"}],
)


async def run(i):
    global ne
    print(i)
    end_at = []
    try:
        end_at += [time.time() + 7]
        response = await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"{i}. Hey, how's it going?"}],
        )
        print(time.time())
        return response
    except ValueError or APIError as e:
        while end_at and end_at[0] < time.time():
            end_at.pop(0)
        if end_at:
            print(time.time(), "sleeping ", end_at[0])
            time.sleep(math.ceil(end_at.pop(0) - time.time()))
        # doens't make requests so don't count against rate limits; run as often as like
        print(e, type(e))
        ne += 1


async def main(data):
    tasks = [run(i) for i in data]
    results = await asyncio.gather(*tasks)
    return results


results = asyncio.run(main(range(6)))

# %%
