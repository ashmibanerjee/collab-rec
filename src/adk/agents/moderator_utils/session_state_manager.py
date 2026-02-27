"""
Session state management for moderator agent.
"""
from typing import Dict, Any, List
import logging

from src.schema.moderator_context import ModeratorContext

logger = logging.getLogger(__name__)


class SessionStateManager:
    """Manages session state updates and context preparation."""

    @staticmethod
    def inject_moderator_context(
        ctx: Any,
        round_number: int,
        collective_offer: List[str],
        rejected_cities_globally: set,
        prev_offers: Dict[str, Dict[str, Any]],
        catalog: List[str] = None
    ) -> None:
        """
        Inject moderator context into session state.

        Args:
            ctx: ADK context
            round_number: Current round number
            collective_offer: Current collective offer
            rejected_cities_globally: Set of globally rejected cities (cumulative)
            prev_offers: Previous offers from agents
            catalog: Full catalog of valid cities (defaults to CITIES from constants)
        """
        ctx.session.state["moderator_round"] = round_number
        ctx.session.state["collective_offer"] = collective_offer
        ctx.session.state["collective_rejection"] = list(rejected_cities_globally)

        # Create agent-specific contexts if not first round
        if prev_offers:
            # Get catalog of all valid cities
            if catalog is None:
                from constants import CITIES
                catalog = CITIES

            # Compute additional_candidates
            collective_set = set(collective_offer) if collective_offer else set()
            rejected_set = set(rejected_cities_globally) if rejected_cities_globally else set()
            additional_candidates = [
                city for city in catalog
                if city not in collective_set and city not in rejected_set
            ]

            ctx.session.state["agent_contexts"] = {
                agent: ModeratorContext(
                    round=round_number,
                    collective_offer=collective_offer,
                    additional_candidates=additional_candidates,
                    collective_rejection=list(rejected_cities_globally),
                    previous_recommendations=data["valid_offers"],
                    agent_feedback=ctx.session.state.get("agent_feedback", {}).get(agent, "")
                ).model_dump()
                for agent, data in prev_offers.items()
            }
        else:
            ctx.session.state["agent_contexts"] = {}

    @staticmethod
    def update_session_state(
        ctx: Any,
        round_number: int,
        collective_offer: List[str],
        rejected_cities_globally: set,
        parsed_responses: Dict[str, Dict[str, Any]],
        agent_feedback: Dict[str, str],
        agent_contexts: Dict[str, ModeratorContext],
        city_scores: Dict[str, float],
        early_stop_thresholds: Dict[float, int | None],
        should_stop: bool = False
    ) -> None:
        """
        Update session state with results from current round.

        Args:
            ctx: ADK context
            round_number: Current round number
            collective_offer: Current collective offer
            rejected_cities_globally: Set of globally rejected cities (cumulative)
            parsed_responses: Parsed responses from agents
            agent_feedback: Feedback for each agent
            agent_contexts: Moderator contexts for next round
            city_scores: Current city scores
            early_stop_thresholds: Early stopping threshold status
            should_stop: Whether to stop after this round
        """
        ctx.session.state.update({
            "round": round_number,
            "collective_offer": collective_offer,
            "collective_rejection": list(rejected_cities_globally),
            "previous_recommendations": parsed_responses,
            "agent_feedback": agent_feedback,
            "agent_contexts": {
                agent: context.model_dump()
                for agent, context in agent_contexts.items()
            },
            "city_scores": city_scores,
            "early_stopping": early_stop_thresholds,
            "should_stop": should_stop,
            "agent_metrics": {
                agent: {
                    "relevance_score": data.get("relevance_score", 0.0),
                    "reliability_score": data.get("reliability_score", 0.0),
                    "hallucination_rate": data.get("hallucination_rate", 0.0),
                    "total_token_count": data.get("total_token_count", 0)
                }
                for agent, data in parsed_responses.items()
            }
        })

    @staticmethod
    def prepare_agent_contexts_for_next_round(
        round_number: int,
        collective_offer: List[str],
        rejected_cities_globally: set,
        parsed_responses: Dict[str, Dict[str, Any]],
        agent_feedback: Dict[str, str],
        catalog: List[str] = None
    ) -> Dict[str, ModeratorContext]:
        """
        Prepare agent-specific moderator contexts for the next round.

        Args:
            round_number: Current round number
            collective_offer: Current collective offer
            rejected_cities_globally: Set of globally rejected cities (cumulative)
            parsed_responses: Parsed responses from agents
            agent_feedback: Feedback for each agent
            catalog: Full catalog of valid cities (defaults to CITIES from constants)

        Returns:
            Dictionary mapping agent names to ModeratorContext objects
        """
        # Get catalog of all valid cities
        if catalog is None:
            from constants import CITIES
            catalog = CITIES

        # Compute additional_candidates: catalog - collective_offer - rejected_cities_globally
        collective_set = set(collective_offer) if collective_offer else set()
        rejected_set = set(rejected_cities_globally) if rejected_cities_globally else set()

        # Additional candidates are all cities not in collective offer and not rejected
        additional_candidates = [
            city for city in catalog
            if city not in collective_set and city not in rejected_set
        ]

        logger.info(
            f"Preparing contexts for round {round_number + 1}: "
            f"collective_offer={len(collective_set)}, "
            f"rejected_globally={len(rejected_set)}, "
            f"additional_candidates={len(additional_candidates)}"
        )

        agent_contexts = {}

        for agent, data in parsed_responses.items():
            context = ModeratorContext(
                round=round_number + 1,  # Next round number
                collective_offer=collective_offer,
                additional_candidates=additional_candidates,
                collective_rejection=list(rejected_cities_globally),
                previous_recommendations=data["valid_offers"],
                agent_feedback=agent_feedback.get(agent, "")
            )

            agent_contexts[agent] = context

            # Debug: Log what's being passed to each agent
            logger.info(f"ModeratorContext for {agent} (round {round_number + 1}):")
            logger.info(f"  - collective_offer: {len(collective_offer) if collective_offer else 0} cities")
            logger.info(f"  - additional_candidates: {len(additional_candidates)} cities")
            logger.info(f"  - collective_rejection (global): {len(rejected_cities_globally)} cities - {sorted(list(rejected_cities_globally))[:10]}...")
            logger.info(f"  - previous_recommendations: {data['valid_offers']}")
            logger.info(f"  - agent_feedback: {agent_feedback.get(agent, '')[:100]}...")

        return agent_contexts

