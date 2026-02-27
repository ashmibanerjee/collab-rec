import sys
from unittest.mock import MagicMock

# Mock google.adk.agents to avoid import errors in local test environment
mock_adk = MagicMock()
sys.modules["google.adk"] = mock_adk
sys.modules["google.adk.agents"] = mock_adk

from src.adk.agents.moderator import ModeratorAgent
from src.adk.agents.moderator_utils import weighted_reliability_score, post_process_response
from constants import ALLOWED_CITIES, CITIESfrom src.adk.agents.moderator_utils import weighted_reliability_score, post_process_response
from constants import ALLOWED_CITIES, CITIES

class MockAgent:
    def __init__(self, name):
        self.name = name
    
    def generate_response(self, prompt):
        return MagicMock(text="Paris | Beautiful city")

class TestModeratorUtils(unittest.TestCase):
    def test_post_process_response(self):
        text = "Paris | Explanation\nLondon | Another explanation"
        cities = post_process_response(text)
        self.assertEqual(cities, ["Paris", "London"])
        
        text_garbage = "I recommend nothing"
        cities = post_process_response(text_garbage)
        self.assertEqual(cities, [])

class TestModeratorAgent(unittest.TestCase):
    def setUp(self):
        self.specialists = MagicMock()
        self.agent1 = MockAgent("Agent1")
        self.agent2 = MockAgent("Agent2")
        self.specialists.sub_agents = [self.agent1, self.agent2]
        self.moderator = ModeratorAgent(self.specialists)
        
    def test_validate(self):
        valid_text = "Paris | Good"
        is_valid, msg, cities = self.moderator.validate(valid_text)
        self.assertTrue(is_valid)
        self.assertIn("Paris", cities)

        invalid_text = "Atlantis | Underwater"
        is_valid, msg, cities = self.moderator.validate(invalid_text)
        self.assertFalse(is_valid) # Atlantis not in CITIES
        self.assertIn("Atlantis", cities)
        self.assertIn("Hallucinations detected", msg)

    def test_run_flow(self):
        # Mock agent responses
        # Agent 1 returns valid city (ensure it is in CITIES list, e.g. Paris)
        self.agent1.generate_response = MagicMock(return_value=MagicMock(text="Paris | Good"))
        # Agent 2 returns valid city (e.g. London)
        self.agent2.generate_response = MagicMock(return_value=MagicMock(text="London | Good"))
        
        input_state = {"loop_index": 0, "context": "Test Context"}
        result = self.moderator.run(input_state)
        
        self.assertIn("collective_offer", result)
        self.assertTrue(len(result["collective_offer"]) > 0)
        self.assertIn("Paris", self.moderator.city_scores)
        self.assertIn("London", self.moderator.city_scores)

if __name__ == '__main__':
    unittest.main()
