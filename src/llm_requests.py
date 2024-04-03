# %%
import importlib

# importlib.reload(litellm)
import os
import asyncio
import time
import math
from collections import deque
import random
import nest_asyncio
import reprlib
from collections import defaultdict


nest_asyncio.apply()

import litellm
from litellm import completion, batch_completion, Router
from litellm.utils import token_counter

# Already set all API's keys in bashrc:
for k in "OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY".split(", "):
    assert k and len(os.environ[k])


class TokenRateLimiter:
    def __init__(self, intervals, request_limits, token_limits):
        """
        Initializes the token-based rate limiter.

        :param intervals: List of intervals (in seconds) for which to limit requests and tokens.
        :param request_limits: List of maximum number of requests allowed per interval.
        :param token_limits: List of maximum number of tokens allowed per interval.
        """
        self.intervals = intervals
        self.request_limits = request_limits
        assert all((i > 0 for i in request_limits))
        self.token_limits = token_limits
        self.request_queues = {interval: deque() for interval in intervals}
        # {interval: {timestamp: total tokens at that timestamp}}
        self.token_dicts = {interval: defaultdict(int) for interval in intervals}
        # {interval: total tokens in interval}
        self.token_totals = {interval: 0 for interval in intervals}

    async def get_request_slot(self, tokens=1):
        """
        Waits until a request slot and enough token capacity are available for all intervals.
        Returns timestamps for the request in each interval for potential rollback.

        :param tokens: The number of tokens this request will consume.
        :return: timestamp when request was made
        """
        while True:
            now = time.time()
            for interval, request_limit, token_limit in zip(
                self.intervals, self.request_limits, self.token_limits
            ):
                request_queue = self.request_queues[interval]
                token_dict = self.token_dicts[interval]

                # Remove timestamps and tokens for requests outside the current interval
                while request_queue and now - request_queue[0] > interval:
                    timestamp = request_queue.popleft()
                    self.token_totals[interval] -= token_dict[timestamp]
                    assert self.token_totals[interval] >= 0, (self.token_totals, token_dict)
                    del token_dict[timestamp]

                # only check token limit if already made at least 1 request
                if len(request_queue) >= request_limit or (
                    len(request_queue) and (self.token_totals[interval] + tokens > token_limit)
                ):
                    await asyncio.sleep(request_queue[0] + interval - now)
                    break
            else:
                # If all intervals are within limits, proceed
                for interval in self.intervals:
                    self.request_queues[interval].append(now)
                    self.token_dicts[interval][now] += tokens
                    self.token_totals[interval] += tokens
                return now

    def rollback_tokens(self, timestamp, num_tokens):
        """
        Roll back tokens for a failed request by dec num tokens sent at that timestamp

        :param timestamp: float of when sent
        :param num_tokens: num_tokens tried to send
        """
        for interval in self.intervals:
            if timestamp in self.token_dicts[interval]:
                self.token_dicts[interval][timestamp] -= num_tokens
                assert self.token_dicts[interval][timestamp] >= 0, self.token_dicts[interval]
                self.token_totals[interval] -= num_tokens
                assert self.token_totals[interval] >= 0, self.token_totals
                assert self.token_totals[interval] == sum(self.token_dicts[interval].values())

    def rollback_request(self, timestamp, num_tokens):
        """
        request filtered by Router so doesn't counter against real limits

        :param timestamp: float of when sent
        :param num_tokens: num_tokens tried to send
        """
        self.rollback_tokens(timestamp, num_tokens)
        for req_q in self.request_queues.values():
            if req_q and req_q[0] <= timestamp:
                req_q.remove(timestamp)


class RouterRateLimiter:
    def __init__(self, router, burst_intervals, rpm_factors, tpm_factors):
        self.router = router
        self.model_limiters = defaultdict(lambda: None)
        model_names = [d["model_name"] for d in router.healthy_deployments]
        for name in model_names:
            # Extract limits over 1 minute for each model
            max_rpm = max(
                (
                    m["litellm_params"].get("rpm", math.inf)
                    for m in router.healthy_deployments
                    if m["model_name"] == name
                )
            )
            max_tpm = max(
                (
                    m["litellm_params"].get("tpm", math.inf)
                    for m in router.healthy_deployments
                    if m["model_name"] == name
                )
            )

            # Calculate adjusted limits for each interval
            def adj_limits(_pm, _factors):
                return [
                    math.ceil((_pm / 60) * interval * factor)
                    for interval, factor in zip(burst_intervals, _factors)
                ]

            request_limits = adj_limits(max_rpm, rpm_factors)
            token_limits = adj_limits(max_tpm, tpm_factors)

            # Initialize and store the TokenRateLimiter for the model
            self.model_limiters[name] = TokenRateLimiter(
                burst_intervals, request_limits, token_limits
            )

    async def make_request(self, *, model, messages, **kwargs):
        """
        Retrieve the TokenRateLimiter instance for a given model name.
        kwargs[_is_test]: flag regardless of bool value. Sleeps 0.5 sec and returns 1

        :param model_name: Name of the model for which to retrieve the rate limiter.
        :return: TokenRateLimiter instance for the model.
        """
        tokens_used = token_counter(messages=messages)
        limiter = self.model_limiters[model]
        i = 0
        while i < 3:
            request_time = await limiter.get_request_slot(tokens_used)
            try:
                if "_is_test" in kwargs:
                    await asyncio.sleep(0.5)
                    return 1
                response = await self.router.acompletion(model, messages, **kwargs)
                return response
            except Exception as e:
                print(f"Error: {type(e)} {model}: {e} {kwargs} {reprlib.repr(messages)}")
                if isinstance(e, ValueError):
                    # Router filtered so doesn't actually count against any limits
                    limiter.rollback_request(request_time, tokens_used)
                    await asyncio.sleep(1.5**i)  # trying again against router
                else:
                    # made request to model, but that doesn't count against TPM?
                    limiter.rollback_tokens(request_time, tokens_used)
                # Other Error types?
                if getattr(e, "status_code", 0) // 100 == 4:
                    print("leaving early")
                    return e
                i += 1
        return None

    def share_rate_limits(self, shared_rate_limits):
        """
        Take dict of model_name: other models names to subsume under
        eg.     shared_rate_limits = {
        "gpt-4-turbo-preview":["gpt-4-0125-preview","gpt-4-1106-preview"]
        }
        combines these 3 rate limits
        """
        for new_l, shared in shared_rate_limits.items():
            for old_l in shared:
                self.model_limiters[old_l] = self.model_limiters[new_l]


class BatchRequests:
    """
    Note: makes 1333 rq/sec or 80k/minute when await asyncio.sleep(0.5)
    20k-1 empty reqs took 76 sec, 6 sec overhead
    """

    BURST_INTERVALS = [0.2, 3.5, 60]  # Intervals for burst limiting in seconds
    BURST_FACTORS = [25, 5, 1]  # Fudge factors for each interval

    def __init__(self, max_tokens=500):
        model_list = [  # list of model deployments
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "max_tokens": max_tokens,
                },
                # "tpm": 2,
                # "rpm": 15000,
                "rpm": 1 * 8,
                "tpm": 30 * 8,
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "max_tokens": max_tokens,
                },
                "rpm": 10000,
                "tpm": 300000,
            },
            # Actually have Shared limits amoung
            # gpt-4-turbo-preview
            # gpt-4-0125-preview
            # gpt-4-1106-preview
            #     "rpm": 10000,
            #    "tpm": 1500000,
            {
                "model_name": "gpt-4-turbo-preview",
                "litellm_params": {
                    "model": "gpt-4-turbo-preview",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "max_tokens": max_tokens,
                },
                "rpm": 10000,
                "tpm": 1500000,
            },
            {
                "model_name": "gpt-4-0125-preview",
                "litellm_params": {
                    "model": "gpt-4-0125-preview",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "max_tokens": max_tokens,
                },
                "rpm": 10000,
                "tpm": 1500000,
            },
            {
                "model_name": "gpt-4-1106-preview",
                "litellm_params": {
                    "model": "gpt-4-1106-preview",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "max_tokens": max_tokens,
                },
                "rpm": 10000,
                "tpm": 1500000,
            },
        ]

        self._router = Router(
            model_list=model_list,
            routing_strategy="usage-based-routing",
            debug_level="DEBUG",
        )
        self.router_rate_limited = RouterRateLimiter(
            self._router, self.BURST_INTERVALS, self.BURST_FACTORS, self.BURST_FACTORS
        )
        self.router_rate_limited.share_rate_limits(
            shared_rate_limits={"gpt-4-turbo-preview": ["gpt-4-0125-preview", "gpt-4-1106-preview"]}
        )

    def batch_router_requests(self, list_of_kwargs):
        return asyncio.run(
            asyncio.gather(
                *[self.router_rate_limited.make_request(**kwargs) for kwargs in list_of_kwargs]
            )
        )


# router2 = Router(
#    model_list=model_list,
#    routing_strategy="usage-based-routing",
#    debug_level="DEBUG",
# )

litellm.set_verbose = False
br = BatchRequests(max_tokens=5)
list_of_req = [
    m
    for n in ["gpt-4-turbo-preview", "gpt-4-1106-preview", "gpt-4-0125-preview"][:1]
    for m in [
        {
            "model": n,
            "messages": [{"role": "user", "content": f"{i}. {random.random()}" * 99999}],
            # "_is_test": True,
        }
        for i in range(1)
    ]
]
print(time.time())
results = br.batch_router_requests(list_of_req)
print(time.time())
results


# %%
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
asyncio.run(does_error(range(3), sync=False))
assert asyncio.run(does_error(range(3), sync=False))
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
            (m["litellm_params"].get("rpm", 0) for m in router.healthy_deployments),
            default=100,
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
                {
                    "role": "user",
                    "content": f"{i}. Which model number are you? {random.random()}",
                }
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

from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
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
                kwargs,
                "stop",
                ["Sorry, ", "I'm sorry", "I apologize", "I'm really sorry"],
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
