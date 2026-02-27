"""
Test scoring manager to verify agent-specific filter behavior.
"""
import unittest
from unittest.mock import MagicMock
from src.adk.agents.moderator_utils.scoring_manager import (
    ScoringManager,
    AGENT_FILTER_MAPPING
)
from adk.agents.moderator_utils.compute_scores import compute_agent_success_scores


class MockRetriever:
    """Mock retriever for testing."""

    def __init__(self, city_data):
        self.city_data = city_data

    def match_city_with_filters(self, city, filters):
        """Return which filters match for a city."""
        if city not in self.city_data:
            return {}

        city_attrs = self.city_data[city]
        matched = {}
        for key, value in filters.items():
            if city_attrs.get(key) == value:
                matched[key] = value
        return matched


class TestScoringManager(unittest.TestCase):

    def setUp(self):
        """Set up test data."""
        # Mock city data
        self.city_data = {
            "Paris": {
                "budget": "moderate",
                "month": "May",
                "interests": "culture",
                "aqi": "good",
                "walkability": "great",
                "seasonality": "high",
                "popularity": "high"
            },
            "Berlin": {
                "budget": "moderate",
                "month": "May",
                "interests": "culture",
                "aqi": "great",
                "walkability": "great",
                "seasonality": "low",
                "popularity": "medium"
            }
        }

        self.retriever = MockRetriever(self.city_data)
        self.catalog = ["Paris", "Berlin", "London", "Rome"]
        self.scoring_manager = ScoringManager(
            k=10,
            retriever=self.retriever
        )

    def test_agent_filter_mapping_exists(self):
        """Verify AGENT_FILTER_MAPPING is properly defined."""
        self.assertIn('personalization_agent', AGENT_FILTER_MAPPING)
        self.assertIn('sustainability_agent', AGENT_FILTER_MAPPING)
        self.assertIn('popularity_agent', AGENT_FILTER_MAPPING)

        # Verify filter keys
        self.assertEqual(
            AGENT_FILTER_MAPPING['personalization_agent'],
            {"budget", "month", "interests"}
        )
        self.assertEqual(
            AGENT_FILTER_MAPPING['sustainability_agent'],
            {"aqi", "walkability", "seasonality"}
        )
        self.assertEqual(
            AGENT_FILTER_MAPPING['popularity_agent'],
            {"popularity"}
        )

    def test_compute_relevance_with_agent_filters(self):
        """Test that relevance is computed correctly with agent-specific filters."""
        # All travel filters
        all_filters = {
            "budget": "moderate",
            "month": "May",
            "interests": "culture",
            "aqi": "good",
            "walkability": "great",
            "seasonality": "high",
            "popularity": "high"
        }

        # Test sustainability agent filters only
        sustainability_filters = {
            key: val for key, val in all_filters.items()
            if key in AGENT_FILTER_MAPPING['sustainability_agent']
        }

        # Paris matches aqi=good, walkability=great, seasonality=high (3/3 = 1.0)
        relevance = self.scoring_manager.compute_relevance(
            offers=["Paris"],
            filters=sustainability_filters
        )
        self.assertEqual(relevance, 1.0)

        # Berlin matches aqi=great (not good), walkability=great, seasonality=low (not high)
        # So only 1/3 filters match
        relevance_berlin = self.scoring_manager.compute_relevance(
            offers=["Berlin"],
            filters=sustainability_filters
        )
        self.assertAlmostEqual(relevance_berlin, 1/3, places=2)

    def test_aggregate_scores_uses_agent_specific_filters(self):
        """Test that aggregate_scores uses agent-specific filters correctly."""
        all_filters = {
            "budget": "moderate",
            "month": "May",
            "interests": "culture",
            "aqi": "good",
            "walkability": "great",
            "seasonality": "high",
            "popularity": "high"
        }

        parsed_responses = {
            'sustainability_agent': {
                'valid_offers': ['Paris'],
                'candidates': ['Paris'],
                'rejections': []
            },
            'personalization_agent': {
                'valid_offers': ['Berlin'],
                'candidates': ['Berlin'],
                'rejections': []
            }
        }

        city_scores = {city: 0.0 for city in self.catalog}
        prev_offers = {}
        collective_offer = []

        # Run aggregate_scores
        updated_scores, hallucinated = self.scoring_manager.aggregate_scores(
            parsed_responses=parsed_responses,
            city_scores=city_scores,
            prev_offers=prev_offers,
            collective_offer=collective_offer,
            travel_filters=all_filters,
            rejected_cities=set(),
            smoothing_weight=None
        )

        # Verify that relevance_score was computed and stored
        self.assertIn('relevance_score', parsed_responses['sustainability_agent'])
        self.assertIn('relevance_score', parsed_responses['personalization_agent'])

        # Sustainability agent should have high relevance (Paris matches all sustainability filters)
        sustainability_relevance = parsed_responses['sustainability_agent']['relevance_score']
        self.assertEqual(sustainability_relevance, 1.0)

        # Personalization agent should have high relevance (Berlin matches all personalization filters)
        personalization_relevance = parsed_responses['personalization_agent']['relevance_score']
        self.assertEqual(personalization_relevance, 1.0)

    def test_moderator_uses_all_filters(self):
        """Test that moderator (non-specialized agent) uses all travel filters."""
        all_filters = {
            "budget": "moderate",
            "month": "May",
            "interests": "culture",
            "aqi": "good",
            "walkability": "great",
            "seasonality": "high",
            "popularity": "high"
        }

        parsed_responses = {
            'moderator': {  # Not in AGENT_FILTER_MAPPING
                'valid_offers': ['Paris'],
                'candidates': ['Paris'],
                'rejections': []
            }
        }

        city_scores = {city: 0.0 for city in self.catalog}
        prev_offers = {}
        collective_offer = []

        # Run aggregate_scores
        updated_scores, hallucinated = self.scoring_manager.aggregate_scores(
            parsed_responses=parsed_responses,
            city_scores=city_scores,
            prev_offers=prev_offers,
            collective_offer=collective_offer,
            travel_filters=all_filters,
            rejected_cities=set(),
            smoothing_weight=None
        )

        # Moderator should have relevance computed against ALL filters
        moderator_relevance = parsed_responses['moderator']['relevance_score']

        # Paris matches all 7 filters (7/7 = 1.0)
        self.assertEqual(moderator_relevance, 1.0)


if __name__ == '__main__':
    unittest.main()

