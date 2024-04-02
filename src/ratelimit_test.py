# %%
import asyncio
import os
import pytest
import random
from typing import Any, Dict, List

from litellm import Router
from pydantic import BaseModel, Field, validator


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


@pytest.fixture(params=[(2000, 50), (2, 50000)])
def router(request) -> Router:
    config: RouterConfig = RouterConfig.parse_obj(request.param)
    model_list = base_model_list.copy()
    model_list[0]["rpm"] = config.rpm
    model_list[0]["tpm"] = config.tpm
    return Router(
        model_list=model_list,
        routing_strategy="usage-based-routing",
        debug_level="DEBUG",
    )


async def arun_doesnt_error(router: Router, i: int) -> Any:
    response = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"{i}. Hey, how's it going? {random.random()}"}],
    )
    return response


@pytest.mark.asyncio
async def test_does_error_sync(router: Router):
    data = range(6)
    results = [arun_doesnt_error(router, i) for i in data]
    assert all(results), "Not all calls were successful"


@pytest.mark.asyncio
async def test_does_error_async(router: Router):
    data = range(6)
    tasks = [arun_doesnt_error(router, i) for i in data]
    results = await asyncio.gather(*tasks)
    assert all(results), "Not all calls were successful"


## %%
# import asyncio
# import os
# import pytest
# import random
# from litellm import Router
#
## Define a single model list and modify as needed
# base_model_list = [
#    {
#        "model_name": "gpt-3.5-turbo",
#        "litellm_params": {
#            "model": "gpt-3.5-turbo",
#            "api_key": os.getenv("OPENAI_API_KEY"),
#            "max_tokens": 5,
#        },
#    }
# ]
#
#
# @pytest.fixture(params=[(2000, 50), (2, 50000)])
# def router(request):
#    rpm, tpm = request.param
#    model_list = base_model_list.copy()
#    model_list[0]["rpm"] = rpm
#    model_list[0]["tpm"] = tpm
#    return Router(
#        model_list=model_list, routing_strategy="usage-based-routing", debug_level="DEBUG"
#    )
#
#
# async def arun_doesnt_error(router, i):
#    response = await router.acompletion(
#        model="gpt-3.5-turbo",
#        messages=[{"role": "user", "content": f"{i}. Hey, how's it going? {random.random()}"}],
#    )
#    return response
#
#
# @pytest.mark.asyncio
# @pytest.mark.parametrize("sync_mode", [True, False])
# async def test_does_error(router, sync_mode):
#    data = range(6)
#    if sync_mode:
#        results = [arun_doesnt_error(router, i) for i in data]
#    else:
#        tasks = [arun_doesnt_error(router, i) for i in data]
#        results = await asyncio.gather(*tasks)
#    assert all(results), "Not all calls were successful"
#
#
## %%
#
