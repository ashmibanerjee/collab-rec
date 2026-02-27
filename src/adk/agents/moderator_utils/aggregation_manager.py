"""
Aggregation utilities for creating collective recommendations.
"""
from typing import Dict, List
import logging

from src.adk.agents.moderator_utils.helpers import zscore_normalize, min_max_normalize

logger = logging.getLogger(__name__)


class AggregationManager:
    """Manages aggregation of scores and creation of collective offers."""

    def __init__(self, k: int = 10, standardize: bool = False):
        """
        Initialize aggregation manager.

        Args:
            k: Number of top recommendations to include
            standardize: Whether to use z-score (True) or min-max (False) normalization
        """
        self.k = k
        self.standardize = standardize

    def normalize_scores(self, city_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize city scores.

        Args:
            city_scores: Dictionary of city scores

        Returns:
            Normalized city scores
        """
        if self.standardize:
            return zscore_normalize(city_scores)
        else:
            return min_max_normalize(city_scores)

    def create_collective_offer(
        self,
        city_scores: Dict[str, float]
    ) -> List[str]:
        """
        Create collective offer from city scores.

        Args:
            city_scores: Dictionary of city scores (should be normalized)

        Returns:
            List of top k cities
        """
        # Sort cities by score in descending order
        ranked = sorted(city_scores.items(), key=lambda x: x[1], reverse=True)

        # Take top k cities
        collective_offer = [city for city, _ in ranked[:self.k]]

        logger.info(f"Created collective offer with {len(collective_offer)} cities")
        logger.debug(f"Top cities: {collective_offer[:5]}")

        return collective_offer

    def get_non_zero_scores(
        self,
        city_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Extract non-zero city scores.

        Args:
            city_scores: Dictionary of city scores

        Returns:
            Dictionary containing only cities with non-zero scores
        """
        return {
            city: score
            for city, score in city_scores.items()
            if score != 0.0
        }

