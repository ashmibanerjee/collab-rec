from typing import List
from fastapi import APIRouter, HTTPException

from src.adk.agents.agent import create_specialized_agent
from src.adk.pipeline import create_pipeline
from src.adk.run import get_model_response
from src.schema.agent_response import AgentResponse
from src.schema.agent_request import AgentRequest

# Create a router for user endpoints
router = APIRouter(tags=["ADK Endpoints"])


@router.post("/personalization_agent", response_model=List[AgentResponse] | List[dict], response_model_exclude_none=False)
async def get_agent_response(request: AgentRequest, model_name: str = "gemini-2.5-flash"):
    try:
        model_init = await create_specialized_agent(
                agent_name='personalization_agent',
                filters=request.filters,
                agent_desc='An agent that generates travel recommendations for city trips, given an user query and '
                           'its associated constraints.', model_name=model_name)
        response = await get_model_response(query=request.query, root_agent=model_init,model_name=model_name)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sustainability_agent", response_model=List[AgentResponse] | List[dict],
             response_model_exclude_none=False)
async def get_agent_response(request: AgentRequest, model_name: str = "gemini-2.5-flash"):
    try:
        model_init = await create_specialized_agent(agent_name='sustainability_agent',
                                                    filters=request.filters,
                                                    agent_desc='An agent that generates travel recommendations for city trips, given an user query, while prioritizing sustainable travel options. ', model_name=model_name)
        response = await get_model_response(query=request.query, root_agent=model_init, model_name=model_name)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/popularity_agent", response_model=List[AgentResponse] | List[dict], response_model_exclude_none=False)
async def get_agent_response(request: AgentRequest, model_name: str = "gemini-2.5-flash"):
    try:
        model_init = await create_specialized_agent(agent_name='popularity_agent',
                                                    agent_desc='An agent that generates travel recommendations for city trips, given an user query, while prioritizing less popular or hidden gems over popular/crowded travel options. ',
                                                    filters=request.filters, model_name=model_name)
        response = await get_model_response(query=request.query, root_agent=model_init, model_name=model_name)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-negotiation-pipeline", response_model=List[AgentResponse] | List[dict], response_model_exclude_none=False)
async def get_pipeline(
    request: AgentRequest,
    rounds: int = 3,
    early_stopping_threshold: float = None,
    min_rounds: int = 1,
    model_name: str = "gemini-2.5-flash"
):
    """
    Run the multi-round negotiation pipeline.

    Args:
        request: Agent request with query and filters
        rounds: Maximum number of rounds (default: 3)
        early_stopping_threshold: Threshold for early stopping (None=MN, 0.2=M20, 0.6=M60)
        min_rounds: Minimum rounds before evaluating early stopping (default: 1)
    """
    try:
        model_init = await create_pipeline(
            request,
            rounds=rounds,
            early_stopping_threshold=early_stopping_threshold,
            min_rounds=min_rounds,
            model_name=model_name
        )
        response = await get_model_response(query=request.query, root_agent=model_init, request=request, model_name=model_name)

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))