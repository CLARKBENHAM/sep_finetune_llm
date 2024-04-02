# %%
import asyncio
import os
import pytest
import random
from typing import Any, Dict, List

from litellm import Router
from pydantic import BaseModel, Field, validator
from litellm import utils

# Define a single model list and modify as needed
base_model_list = [
    {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {
            "model": "gpt-3.5-turbo",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "max_tokens": 5,
        },
    }
]


class RouterConfig(BaseModel):
    rpm: int
    tpm: int


@pytest.fixture
def router_factory():
    def create_router(rpm, tpm):
        model_list = base_model_list.copy()
        model_list[0]["rpm"] = rpm
        model_list[0]["tpm"] = tpm
        return Router(
            model_list=model_list,
            routing_strategy="usage-based-routing",
            debug_level="DEBUG",
        )

    return create_router


def generate_messages(num_messages):
    messages = []
    for i in range(num_messages):
        messages.append({"role": "user", "content": f"{i}. Hey, how's it going? {random.random()}"})
    return messages


def calculate_limits(messages):
    rpm = len(messages)
    tpm = utils.token_counter(messages=messages) + rpm * 5  # for completion limits
    # Implement your logic to calculate rpm and tpm based on total_tokens
    # ...
    return rpm, tpm


async def async_call(router: Router, messages) -> Any:
    tasks = [router.acompletion(model="gpt-3.5-turbo", messages=messages)]
    return await asyncio.gather(*tasks)


def sync_call(router: Router, messages) -> Any:
    return [router.completion(model="gpt-3.5-turbo", messages=messages)]


class ExpectNoException(Exception):
    pass


@pytest.mark.parametrize(
    "num_messages, num_rate_limits",
    [
        (2, 3),  # No exception expected
        (6, 3),  # Expect ValueError
    ],
)
@pytest.mark.parametrize("sync_mode", [True, False])  # Use parametrization for sync/async
def test_rate_limit(router_factory, num_messages, num_rate_limits, sync_mode):
    messages = generate_messages(max(num_messages, num_rate_limits))
    rpm, tpm = calculate_limits(messages)
    router = router_factory(rpm, tpm)
    messages = messages[:num_messages]
    expected_exception = ExpectNoException if num_messages <= num_rate_limits else ValueError

    with pytest.raises(expected_exception) as excinfo:
        if sync_mode:
            results = sync_call(router, messages)
        else:
            results = asyncio.run(async_call(router, messages))
        raise ExpectNoException

    print(results)
    if expected_exception is not ExpectNoException:
        assert "No deployments available for selected model" in str(excinfo.value)
    else:
        assert len([i for i in results if i is not None]) == num_messages


num_messages, num_rate_limits = 2, 3
generate_messages(max(num_messages, num_rate_limits))
# %%
# import asyncio
# import os
# import pytest
# import random
# from typing import Any, Dict, List
#
# from litellm import Router
# from pydantic import BaseModel, Field, validator
#
#
# # Define a single model list and modify as needed
# base_model_list = [
#     {
#         "model_name": "gpt-3.5-turbo",
#         "litellm_params": {
#             "model": "gpt-3.5-turbo",
#             "api_key": os.getenv("OPENAI_API_KEY"),
#             "max_tokens": 5,
#         },
#     }
# ]
#
#
# class RouterConfig(BaseModel):
#     rpm: int
#     tpm: int
#
#
# async def async_call(router: Router, data) -> Any:
#     tasks = [
#         router.acompletion(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": f"{i}. Hey, how's it going? {random.random()}"}],
#         )
#         for i in data
#     ]
#     return await asyncio.gather(*tasks)
#
#
# def sync_call(router: Router, data) -> Any:
#     return [
#         router.completion(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": f"{i}. Hey, how's it going? {random.random()}"}],
#         )
#         for i in data
#     ]
#
#
# @pytest.fixture
# def router_factory():
#     def create_router(rpm, tpm):
#         model_list = base_model_list.copy()
#         model_list[0]["rpm"] = rpm
#         model_list[0]["tpm"] = tpm
#         return Router(
#             model_list=model_list,
#             routing_strategy="usage-based-routing",
#             debug_level="DEBUG",
#         )
#
#     return create_router
#
#
# @pytest.mark.parametrize("sync_mode", [True, False])  # Use parametrization for sync/async
# @pytest.mark.parametrize("rpm, tpm", [(100, 55), (2, 100)])
# def test_runs(router_factory, rpm, tpm, sync_mode):
#     router = router_factory(rpm, tpm)
#     data = range(2)  # 47 tokens
#     if sync_mode:
#         results = sync_call(router, data)  # No need for asyncio.run for sync calls
#     else:
#         results = asyncio.run(async_call(router, data))  # Run async test in event loop
#     assert len(results) == 2
#
#
# @pytest.mark.parametrize("sync_mode", [True, False])  # Use parametrization for sync/async
# @pytest.mark.parametrize("rpm, tpm", [(100, 55), (2, 100)])
# def test_will_ratelimit(router_factory, rpm, tpm, sync_mode):
#     router = router_factory(rpm, tpm)
#     data = range(3)
#     with pytest.raises(Exception) as excinfo:
#         if sync_mode:
#             print(sync_call(router, data))  # No need for asyncio.run for sync calls
#         else:
#             asyncio.run(async_call(router, data))  # Run async test in event loop
#     assert "No deployments available for selected model" in str(excinfo.value)
