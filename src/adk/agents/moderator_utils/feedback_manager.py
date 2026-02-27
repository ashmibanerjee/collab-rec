"""
Feedback generation for agents based on their performance.
"""
from typing import Dict, Any
import logging

from src.adk.agents.moderator_utils.helpers import (
    get_feedback_text,
    get_hallucination_feedback,
    get_rejection_feedback
)

logger = logging.getLogger(__name__)


class FeedbackManager:
    """Manages feedback generation for agents."""

    def __init__(self, k: int = 10):
        """
        Initialize feedback manager.

        Args:
            k: Number of top recommendations to consider
        """
        self.k = k

    def generate_agent_feedback(
        self,
        agent: str,
        agent_data: Dict[str, Any],
        collective_offer: list,
        compute_hallucination_fn
    ) -> str:
        """
        Generate feedback for a single agent.

        Args:
            agent: Agent identifier
            agent_data: Parsed data for this agent
            collective_offer: Current collective offer
            compute_hallucination_fn: Function to compute hallucination

        Returns:
            Feedback text for the agent
        """
        # Calculate proportion of agent's offers in collective offer
        valid_offers = agent_data["valid_offers"]
        common_cities = set(valid_offers) & set(collective_offer)
        proportion = len(common_cities) / self.k if self.k > 0 else 0.0

        # Build feedback message
        feedback = get_feedback_text(proportion, self.k)

        # Add hallucination feedback
        hallucination_rate = compute_hallucination_fn(agent_data["candidates"])
        feedback += get_hallucination_feedback(hallucination_rate, self.k)

        # Add rejection feedback
        num_rejections = len(agent_data["rejections"])
        feedback += get_rejection_feedback(num_rejections, self.k)

        return feedback

    def generate_all_feedback(
        self,
        parsed_responses: Dict[str, Dict[str, Any]],
        collective_offer: list,
        compute_hallucination_fn
    ) -> Dict[str, str]:
        """
        Generate feedback for all agents.

        Args:
            parsed_responses: Parsed responses from all agents
            collective_offer: Current collective offer
            compute_hallucination_fn: Function to compute hallucination

        Returns:
            Dictionary mapping agent names to feedback text
        """
        agent_feedback = {}

        for agent, data in parsed_responses.items():
            feedback = self.generate_agent_feedback(
                agent, data, collective_offer, compute_hallucination_fn
            )
            agent_feedback[agent] = feedback
            logger.debug(f"Generated feedback for {agent}: {feedback[:100]}...")

        return agent_feedback

