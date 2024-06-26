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
import contextlib
from datetime import datetime

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
                    assert self.token_totals[interval] >= 0, (
                        self.token_totals,
                        token_dict,
                    )
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

    @contextlib.asynccontextmanager
    async def rate_limited_request(self, model, tokens_used):
        """Context manager to handle rate limiting for a request."""
        assert (
            model in self.model_limiters
        ), f"unexpected model name, {model} not in {self.model_limiters}"
        limiter = self.model_limiters[model]
        request_time = await limiter.get_request_slot(tokens_used)
        try:
            yield request_time
        except Exception as e:
            print(f"Error: {type(e)} {model}: {e}")
            if isinstance(e, ValueError):
                # Router filtered so doesn't actually count against any limits
                limiter.rollback_request(request_time, tokens_used)
            else:
                # made request to model, but that doesn't count against TPM?
                limiter.rollback_tokens(request_time, tokens_used)
            raise

    async def make_request(self, model, messages=None, **kwargs):
        """
        Make a request with rate limiting using the context manager.
        """
        tokens_used = token_counter(messages=messages, text=kwargs.get("input", None))
        for i in range(3):
            async with self.rate_limited_request(model, tokens_used) as request_time:
                try:
                    if "_is_test" in kwargs:
                        await asyncio.sleep(0.5)
                        return 1
                    if "sync" in kwargs:
                        del kwargs["sync"]
                        response = self.router.completion(model, messages, **kwargs)
                    elif "input" in kwargs:
                        assert "text-moderation" in model
                        # passes in input
                        response = await self.router.amoderation(model=model, **kwargs)
                    else:
                        response = await self.router.acompletion(model, messages, **kwargs)
                    return response
                except Exception as e:
                    print(
                        f"Error: {type(e)} {model} {datetime.fromtimestamp(request_time)}:"
                        f" {e} {kwargs} {reprlib.repr(messages)}"
                    )
                    # Issue with malformed Request (Are there other Error types this misses?)
                    if (
                        isinstance(e, litellm.exceptions.BadRequestError)
                        or isinstance(e, TypeError)
                        or getattr(e, "status_code", 0) // 100 == 4
                    ):
                        print(f"leaving early: {type(e)} {e}")
                        return e
                    # else retry
                    await asyncio.sleep(1.5**i)
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
        # max_tokens should be max_output_tokens but that's not always defined
        model_list = [  # list of model deployments
            #### OpenAI tier 5 https://platform.openai.com/account/limits
            # gpt-3.5
            *[
                {
                    "model_name": n,
                    "litellm_params": {  # params for litellm completion/embedding call
                        "model": n,
                        "api_key": os.getenv("OPENAI_API_KEY"),
                        "max_tokens": max_tokens,
                    },
                    "tpm": 2e6,
                    "rpm": 1e4,
                }
                for n in [
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-0125",
                    "gpt-3.5-turbo-0301",
                    "gpt-3.5-turbo-0613",
                    "gpt-3.5-turbo-1106",
                ]
            ],
            # embeddings
            *[
                {
                    "model_name": n,
                    "litellm_params": {
                        "model": n,
                        "api_key": os.getenv("OPENAI_API_KEY"),
                    },
                    "tpm": 2e6,
                    "rpm": 1e4,
                }
                for n in [
                    "text-embedding-3-large",
                    "text-embedding-3-small",
                    "text-embedding-ada-002",
                ]
            ],
            # gpt-4
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "max_tokens": max_tokens,
                },
                "rpm": 1e4,
                "tpm": 3e5,
            },
            # Actually have Shared limits amoung
            # gpt-4-turbo-preview
            # gpt-4-0125-preview
            # gpt-4-1106-preview
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
            # text moderation
            *[
                {
                    "model_name": n,
                    "litellm_params": {
                        "model": n,
                        "api_key": os.getenv("OPENAI_API_KEY"),
                    },
                    "tpm": 1.5e5,
                    "rpm": 1e3,
                }
                for n in [
                    "text-moderation-latest",
                    "text-moderation-stable",
                ]
            ],
            #### Anthropic
            # build tier-2 (not sure which teir we're on) https://docs.anthropic.com/claude/reference/rate-limits
            #{
            #    "model_name": "claude-3-haiku-20240307",
            #    "litellm_params": {
            #        "model": "anthropic/claude-3-haiku-20240307",
            #        "api_key": os.getenv("ANTHROPIC_API_KEY"),
            #        "max_tokens": max_tokens,
            #    },
            #    "tpm": 1e5,
            #    "rpm": 1e3,
            #},
            #{
            #    "model_name": "claude-3-sonnet-2024029",
            #    "litellm_params": {
            #        "model": "anthropic/claude-3-sonnet-2024029",
            #        "api_key": os.getenv("ANTHROPIC_API_KEY"),
            #        "max_tokens": max_tokens,
            #    },
            #    "tpm": 8e4,
            #    "rpm": 1e3,
            #},
            # Opus not hosted
            {
                "model_name": "claude-3-opus-20240307",
                "litellm_params": {
                    "model": "anthropic/claude-3-opus-20240307",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    "max_output_tokens": max_tokens,
                },
                "tpm": 4e4,
                "rpm": 1e3,
            },
            # Use google hosting, have free credits for now
            {
                "model_name": "claude-3-haiku-20240307",
                "litellm_params": {
                    "model": "vertex_ai/claude-3-haiku@20240307",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    # "max_output_tokens": max_tokens,
                },
                "tpm": 1e5,
                "rpm": 10,  # per region, so could suffle that?
            },
            {  # note this date isn't the same as anthropics
                "model_name": "claude-3-sonnet-20240229",
                "litellm_params": {
                    "model": "vertex_ai/claude-3-sonnet@20240229",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    # "max_output_tokens": max_tokens,
                },
                "tpm": 8e4,
                "rpm": 10,
            },
            #### Gemini
            # pro
            {
                "model_name": "gemini-pro",
                "litellm_params": {
                    "model": "vertex_ai/gemini-pro",
                    "api_key": os.getenv("GEMINI_API_KEY"),
                    # "max_output_tokens": max_tokens,
                },
                "tpm": 1e6,  # don't actually know
                "rpm": 300,
            },
            # 1.5 not availible yet
            {
                "model_name": "gemini-pro-1.5",
                "litellm_params": {
                    # "model": "vertex_ai/gemini-pro-1.5", # didnt work
                    "model": "gemini/gemini-1.5-pro-latest",
                    "api_key": os.getenv("GEMINI_API_KEY"),
                    # "max_output_tokens": max_tokens,
                },
                "tpm": 1e6,  # don't actually know
                "rpm": 5,
            },
            # Updated PalM2 chat-bison@002
            {
                "model_name": "chat-bison@002",
                "litellm_params": {
                    "model": "vertex_ai/chat-bison@002",
                    "api_key": os.getenv("GEMINI_API_KEY"),
                    # "max_output_tokens": max_tokens,
                },
                "tpm": 1e6,  # don't actually know
                "rpm": 300,
            },
            ####
        ]

        self._router = Router(
            model_list=model_list,
            routing_strategy="usage-based-routing",
            debug_level="DEBUG",
            enable_pre_call_checks=True,
        )
        self.router_rate_limited = RouterRateLimiter(
            self._router, self.BURST_INTERVALS, self.BURST_FACTORS, self.BURST_FACTORS
        )
        self.router_rate_limited.share_rate_limits(
            shared_rate_limits={"gpt-4-turbo-preview": ["gpt-4-0125-preview", "gpt-4-1106-preview"]}
        )

    def hack_kwargs(self, **kwargs):
        kwargs["stop"] = getattr(
            kwargs, "stop", ["Sorry, ", "I'm sorry", "I apologize", "I'm really sorry"]
        )
        if "stop" in kwargs and (
            "gemini" in kwargs["model"]
            or "chat-bison" in kwargs["model"]
            # temp while using vertex_ai for anthropic hosting
            or "claude-3-haiku" in kwargs["model"]
            or "claude-3-sonnet" in kwargs["model"]
        ):
            del kwargs["stop"]
        # temp while using vertex_ai for anthropic hosting
        if "claude-3-haiku" in kwargs["model"] or "claude-3-sonnet" in kwargs["model"]:
            kwargs["vertex_location"] = "us-central1"
            kwargs["vertex_project"] = "crypto-visitor-419221"
            # kwargs["vertex_publisher"] = "anthropic"
        return kwargs

    def router_request(self, **kwargs):
        return self.router_rate_limited.make_request(**self.hack_kwargs(**kwargs))

    def router_moderation(self, model, input):
        return self.router_rate_limited.amoderation(model=model, input=input)

    def batch_router_requests(self, list_of_kwargs):
        return asyncio.run(
            asyncio.gather(
                *[
                    self.router_rate_limited.make_request(**self.hack_kwargs(**kwargs))
                    for kwargs in list_of_kwargs
                ]
            )
        )


if __name__ == "__main__":
    # litellm.set_verbose = True
    litellm.set_verbose = False
    br = BatchRequests(max_tokens=5)
    list_of_req = [
        m
        # for n in ["gpt-4-turbo-preview", "gpt-4-1106-preview", "gpt-4-0125-preview"] # rate limits work
        # for n in ["gpt-4-turbo-preview", "gemini-pro", "claude-3-haiku-20240307", "chat-bison@002"]
        # for n in ["claude-3-haiku-20240307"]
        for n in ["text-moderation-stable"]
        for m in [
            {
                "sync": True,  # also fix completion, but wait for this
                "model": n,
                # "messages": [{"role": "user", "content": f"{i}. {random.random()}" * 99999}],
                "messages": [{"role": "user", "content": f"{i}. {random.random()}"}],
                "stop": ["Sorry, ", "I'm sorry", "I apologize", "I'm really sorry"],
                # "kwargs": {"stop": ["Sorry, ", "I'm sorry", "I apologize", "I'm really sorry"]},
                # "_is_test": True,
            }
            for i in range(1)
        ]
    ]
    print(time.time())
    results = br.batch_router_requests(list_of_req)
    print(time.time())
    print(results)
