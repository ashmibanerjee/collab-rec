import asyncio

from adk.agents.agent import create_specialized_agent
from adk.run import get_model_response
from schema.agent_request import AgentRequest


async def run_sasi(request: AgentRequest,
                   model_name: str = "gemini-2.5-flash",
                   ):
    model_init = await create_specialized_agent(
        agent_name='sasi_agent',
        agent_desc='An agent that generates travel recommendations for city trips, given an user query, '
                   'while prioritizing sustainable travel options. ',
        filters=request.filters,
        model_name=model_name
    )
    response = await get_model_response(
        query=request.query,
        root_agent=model_init,
        request=request,
        model_name=model_name
    )
    return response
