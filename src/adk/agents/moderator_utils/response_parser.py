"""
Response parsing utilities for agent responses.
"""
import json
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ResponseParser:
    """Handles parsing of agent responses."""

    @staticmethod
    def parse_agent_response(text: str) -> Dict[str, Any]:
        """
        Parse agent response text to extract city recommendations and metadata.

        Args:
            text: Raw response text from agent

        Returns:
            Dictionary with 'candidates' and 'feedback_acknowledged' keys
        """
        try:
            response_text = text.strip()
            response_text = response_text.replace("`", '')
            response_text = response_text.replace("\n", '')
            response_text = response_text.replace("json", '')

            data = json.loads(response_text)

            # Handle dict responses
            if isinstance(data, dict):
                candidates = None

                # Try different possible keys for recommendations
                for key in ["candidates", "recommendation", "cities"]:
                    if key in data:
                        candidates = data[key]
                        break

                # Extract feedback_acknowledged field
                feedback_acknowledged = data.get("feedback_acknowledged", None)

                return {
                    "candidates": candidates or [],
                    "feedback_acknowledged": feedback_acknowledged
                }

            # Handle list responses
            elif isinstance(data, list):
                return {
                    "candidates": data,
                    "feedback_acknowledged": None
                }

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse agent response: {e}")
            logger.debug(f"Response text: {text[:200]}...")

        return {
            "candidates": [],
            "feedback_acknowledged": None
        }

    @staticmethod
    def parse_all_responses(
        raw_responses: Dict[str, Dict[str, Any]],
        city_scores: Dict[str, float],
        rejected_cities_globally: set
    ) -> Dict[str, Dict[str, Any]]:
        """
        Parse all agent responses and extract valid offers.

        Args:
            raw_responses: Raw responses from all agents
            city_scores: Current city scores dictionary
            rejected_cities_globally: Set of globally rejected cities (cumulative across all rounds)

        Returns:
            Parsed responses with valid offers and rejections
        """
        parsed = {}

        for agent, r in raw_responses.items():
            candidates = r.get("candidates", [])
            feedback_acknowledged = r.get("feedback_acknowledged", None)
            total_token_count = r.get("total_token_count", None)

            if not candidates:
                logger.warning(f"Agent {agent} returned no candidates")

            # Extract city names from candidate dictionaries
            city_names = []
            for candidate in candidates:
                if isinstance(candidate, dict) and "city" in candidate:
                    city_name = candidate["city"]
                    city_names.append(city_name)

                    # Initialize new cities with score 0
                    if city_name not in city_scores:
                        city_scores[city_name] = 0.0

                elif isinstance(candidate, str):
                    # Fallback: if candidate is already a string
                    city_names.append(candidate)
                    if candidate not in city_scores:
                        city_scores[candidate] = 0.0

            parsed[agent] = {
                "candidates": city_names,
                "valid_offers": [
                    c for c in city_names
                    if c in city_scores and c not in rejected_cities_globally
                ],
                "rejections": [],
                "feedback_acknowledged": feedback_acknowledged,
                "total_token_count": total_token_count
            }

            logger.info(
                f"Agent {agent}: {len(city_names)} cities parsed, "
                f"{len(parsed[agent]['valid_offers'])} valid offers, "
                f"feedback_acknowledged={feedback_acknowledged}, "
                f"tokens={total_token_count}"
            )

        return parsed

