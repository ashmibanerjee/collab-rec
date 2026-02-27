import asyncio

from google.adk.agents import ParallelAgent, LoopAgent, SequentialAgent
import os
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv

from adk.agents.gptParallelAgents import GPTParallelAgent
from src.adk.agents.agent import create_specialized_agent
from src.adk.agents.moderator import ModeratorAgent
from src.schema.agent_request import AgentRequest
from src.k_base.context_retrieval import ContextRetrieval

load_dotenv()

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "../../prompts/")
ENV = Environment(loader=FileSystemLoader(PROMPT_DIR))


async def create_pipeline(request: AgentRequest = None, rounds: int = 3, early_stopping_threshold: float = None,
                          temperature: float = 0.5,
                          min_rounds: int = 1, model_name: str = "gemini-2.5-flash",
                          rejection_strategy: str = "majority",
                          ablated_component: str | None = None) -> LoopAgent:
    """
    Create the collaborative recommendation pipeline.

    Args:
        request: Agent request with filters
        rounds: Maximum number of rounds (T)
        early_stopping_threshold: Threshold for early stopping (None=MN)
        temperature: Temperature setting for the model
        min_rounds: Minimum rounds before evaluating early stopping (default: 1)
    """
    print(f"[System] Initializing pipeline with model={model_name}, rounds={rounds}, early_stopping_threshold={early_stopping_threshold}, temperature={temperature}, min_rounds={min_rounds}, rejection_strategy={rejection_strategy}, ablated_component={ablated_component}")
    parallel_agents = await create_specialized_agents(request, model_name=model_name, temperature=temperature)

    # Extract filters from request to pass to ModeratorAgent
    travel_filters = request.filters.model_dump(exclude_none=True) if request and request.filters else {}

    # Initialize retriever for grounded relevance scoring
    retriever = ContextRetrieval()

    # Create ModeratorAgent with filters, retriever, and request
    moderator_agent = ModeratorAgent(
        parallel_agent=parallel_agents,
        travel_filters=travel_filters,
        retriever=retriever,  # Pass retriever for relevance scoring
        request=request,  # Pass request so agents can be recreated with feedback
        early_stopping_threshold=early_stopping_threshold,  # Pass early stopping configuration
        min_rounds=min_rounds,  # Pass minimum rounds for early stopping evaluation
        rejection_method=rejection_strategy,
        ablated_component=ablated_component
    )

    # 4. Wrap the sequence in the LoopAgent
    overall_workflow = LoopAgent(
        name="CollabREC_Pipeline",
        sub_agents=[moderator_agent],
        description="Coordinates multi-round negotiation.",
        max_iterations=rounds
    )

    return overall_workflow


async def create_specialized_agents(request: AgentRequest | None,
                                    model_name: str = "gemini-2.5-flash",
                                    temperature: float = 0.5) -> ParallelAgent | GPTParallelAgent:
    """Initialize and return the root agent pipeline."""
    if not model_name:
        raise ValueError("Model name must be provided to create agents.")

    moderator_context = request.moderator_context if request else None

    # Create all agents concurrently
    personalization_agent, sustainability_agent, popularity_agent = await asyncio.gather(
        create_specialized_agent(
            agent_name='personalization_agent',
            query=request.query if request else None,
            model_name=model_name,
            temperature=temperature,
            filters=request.filters,
            moderator_context=moderator_context,
            agent_desc='An agent that generates travel recommendations for city trips, given an user query and its associated constraints.'),
        create_specialized_agent(
            agent_name='sustainability_agent',
            query=request.query if request else None,
            model_name=model_name,
            temperature=temperature,
            filters=request.filters,
            moderator_context=moderator_context,
            agent_desc='An agent that generates travel recommendations for city trips, given an user query, while prioritizing sustainable travel options. '),
        create_specialized_agent(
            agent_name='popularity_agent',
            query=request.query if request else None,
            model_name=model_name,
            temperature=temperature,
            filters=request.filters,
            moderator_context=moderator_context,
            agent_desc='An agent that generates travel recommendations for city trips, given an user query, while prioritizing less popular or hidden gems over popular/crowded travel options. ')
    )
    # 2. Logic Selection: ParallelAgent for Gemini, GPTParallelAgent for OpenAI
    if "gemini" in model_name:
        return ParallelAgent(
            name="ParallelRecAgents",
            sub_agents=[personalization_agent, sustainability_agent, popularity_agent],
            description="Native parallel execution for Gemini."
        )
    else:
        # For GPT/LiteLLM models
        print(f"[System] Using GPTParallelAgent workaround for {model_name}")
        return GPTParallelAgent(
            name="ParallelRecAgents",
            sub_agents=[personalization_agent, sustainability_agent, popularity_agent],
            description="Pseudo-parallel execution with history isolation for GPT."
        )


async def get_root_agent():
    """Async wrapper to get the root agent."""
    return await create_pipeline()
