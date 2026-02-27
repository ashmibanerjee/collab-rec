from pathlib import Path

import asyncio
from adk.pipeline import create_pipeline
from adk.run import get_model_response
from schema.agent_response import AgentResponse
from src.schema.agent_request import AgentRequest


async def run_mami(min_rounds: int,
                   model_name: str,
                   temperature: float,
                   rejection_strategy: str,
                   request: AgentRequest, rounds: int, ablated_component: str | None = None) -> \
list[AgentResponse]:
    model_init = await create_pipeline(
        request,
        model_name=model_name,
        temperature=temperature,
        rounds=rounds,
        early_stopping_threshold=None,
        min_rounds=min_rounds,
        rejection_strategy=rejection_strategy,
        ablated_component=ablated_component
    )
    response = await get_model_response(
        query=request.query,
        root_agent=model_init,
        request=request,
        model_name=model_name
    )
    return response
