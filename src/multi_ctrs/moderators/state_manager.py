import json
import os

class StateManager:
    """
    StateManager is part of the Moderator class and handles tracking of negotiation states 
    over multiple rounds. Each round maintains a dictionary where each item_id has:
    
    - relevance_score: float
    - reliability_score: float
    - hallucination_rate: float
    - offers: list of candidates
    - rejections: list of candidates
    - city_scores: list of floats

    All values are optional and can be updated incrementally. The state can be exported
    to a JSON file per round or for all rounds.
    """

    def __init__(self):
        self.rounds = {}  # {round_number: {item_id: {metrics}}}

    def _get_or_create_item_state(self, round_number, item_id):
        if round_number not in self.rounds:
            self.rounds[round_number] = {}
        if item_id not in self.rounds[round_number]:
            self.rounds[round_number][item_id] = {
                "relevance_score": None,
                "reliability_score": None,
                "hallucination_rate": None,
                "offers": [],
                "rejections": [],
                "city_scores": [], 
                "early_stopping": None
            }
        return self.rounds[round_number][item_id]

    # --- Update methods ---
    def update_relevance_score(self, round_number, item_id, score):
        item_state = self._get_or_create_item_state(round_number, item_id)
        item_state["relevance_score"] = score

    def update_reliability_score(self, round_number, item_id, score):
        item_state = self._get_or_create_item_state(round_number, item_id)
        item_state["reliability_score"] = score

    def update_hallucination_rate(self, round_number, item_id, rate):
        item_state = self._get_or_create_item_state(round_number, item_id)
        item_state["hallucination_rate"] = rate

    def update_offers(self, round_number, item_id, candidates):
        item_state = self._get_or_create_item_state(round_number, item_id)
        item_state["offers"] = candidates

    def update_rejections(self, round_number, item_id, candidates):
        item_state = self._get_or_create_item_state(round_number, item_id)
        item_state["rejections"] = candidates

    def update_city_scores(self, round_number, item_id, scores):
        item_state = self._get_or_create_item_state(round_number, item_id)
        item_state["city_scores"] = scores

    # --- Read methods ---
    def read_round(self, round_number):
        return self.rounds.get(round_number, {})

    def read_all_rounds(self):
        return self.rounds

    def get_item(self, round_number, item_id):
        return self.rounds.get(round_number, {}).get(item_id, {})

    def get_offers_by_item(self, round_number, item_id):
        return self.rounds.get(round_number, {}).get(item_id, {}).get("offers", [])

    def get_all_agent_offers(self, item_id):
        """
        Gathers all items where agent_1 made offers across all rounds.
        Assumes 'agent_1' is an item_id.
        """
        offers = []
        for round_data in self.rounds.values():
            if item_id in round_data:
                offers.extend(round_data[item_id].get("offers", []))

        return list(set(offers))

    # --- Insert whole round ---
    def insert_round_state(self, round_number, round_dict):
        """
        Insert a complete state for a specific round. Overwrites existing round state.
        Ensures all keys are present in each item.
        """
        default_item_state = {
            "relevance_score": None,
            "reliability_score": None,
            "hallucination_rate": None,
            "offers": [],
            "rejections": [],
            "city_scores": [], 
            "early_stopping": None
        }

        full_round_state = {}
        for item_id, item_data in round_dict.items():
            # Fill in missing fields with defaults
            full_item_state = {**default_item_state, **item_data}
            full_round_state[item_id] = full_item_state
        self.rounds[round_number] = full_round_state
        print(f"Updated state manager for round {round_number}")

    # --- Export methods ---
    def export_state_to_json(self, filepath, round_number=None):
        """
        Export state to a JSON file. If round_number is None, export all rounds.
        """
        if round_number is not None:
            data = {round_number: self.rounds.get(round_number, {})}
        else:
            data = self.rounds

        # os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # print(data)

        with open(filepath, "w+") as f:
            json.dump(data, f, indent=4)
        print(f"Exported state to {filepath}")
