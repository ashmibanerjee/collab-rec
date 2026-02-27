"""
Agent recreation utilities for updating agents with feedback.
"""
import logging
from typing import Dict, Any
from google.adk.agents import ParallelAgent

logger = logging.getLogger(__name__)


class AgentRecreationManager:
    """Manages recreation of agents with updated moderator context."""

    def __init__(self, agent_name_mapping: Dict[str, tuple] = None):
        """
        Initialize agent recreation manager.

        Args:
            agent_name_mapping: Mapping of agent names to (name, description) tuples
        """
        self.agent_name_mapping = agent_name_mapping or {
            "personalization_agent": (
                "personalization_agent",
                "An agent that generates travel recommendations for city trips, "
                "given an user query and its associated constraints."
            ),
            "sustainability_agent": (
                "sustainability_agent",
                "An agent that generates travel recommendations for city trips, "
                "given an user query, while prioritizing sustainable travel options."
            ),
            "popularity_agent": (
                "popularity_agent",
                "An agent that generates travel recommendations for city trips, "
                "given an user query, while prioritizing less popular or hidden gems "
                "over popular/crowded travel options."
            ),
        }

    async def recreate_parallel_agents_with_feedback(
        self,
        agent_contexts: Dict[str, Any],
        request: Any
    ) -> ParallelAgent:
        """
        Recreate parallel agents with updated moderator context/feedback.

        Args:
            agent_contexts: Dictionary of ModeratorContext objects (or dicts) for each agent
            request: Original request containing filters

        Returns:
            New ParallelAgent with updated agents
        """
        from src.adk.agents.agent import create_specialized_agent
        from src.schema.moderator_context import ModeratorContext

        logger.info("Recreating parallel agents with feedback...")

        # Get filters from request
        filters = request.filters if request else None

        # Create new agents with moderator context
        new_agents = []
        for agent_name, (name, desc) in self.agent_name_mapping.items():
            # Get agent-specific context if available
            context = agent_contexts.get(agent_name)

            # Handle both ModeratorContext objects and dicts
            if context:
                if isinstance(context, ModeratorContext):
                    moderator_context = context
                elif isinstance(context, dict):
                    moderator_context = ModeratorContext(**context)
                else:
                    logger.warning(f"Unexpected context type for {agent_name}: {type(context)}, skipping")
                    moderator_context = None
            else:
                moderator_context = None

            agent = await create_specialized_agent(
                agent_name=name,
                query=request.query if request else None,
                filters=filters,
                moderator_context=moderator_context,
                agent_desc=desc
            )
            new_agents.append(agent)

            if moderator_context:
                logger.info(
                    f"Created {agent_name} with feedback: "
                    f"{moderator_context.agent_feedback[:50]}..."
                )
                logger.info(
                    f"  ModeratorContext details for {agent_name}:"
                )
                logger.info(f"    - Round: {moderator_context.round}")
                logger.info(f"    - Collective offer: {moderator_context.collective_offer}")
                logger.info(f"    - Additional candidates: {len(moderator_context.additional_candidates) if moderator_context.additional_candidates else 0} cities")
                logger.info(f"    - Collective rejection: {moderator_context.collective_rejection}")
                logger.info(f"    - Previous recommendations: {moderator_context.previous_recommendations}")
            else:
                logger.info(f"Created {agent_name} without feedback (first round)")

        # Create new ParallelAgent with updated agents
        new_parallel_agent = ParallelAgent(
            name="ParallelRecAgents",
            sub_agents=new_agents,
            description="Runs multiple agents in parallel to gather information."
        )

        return new_parallel_agent

