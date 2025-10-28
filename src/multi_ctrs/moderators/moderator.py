import json, re
import numpy as np
from typing import List, Dict, Any, Callable
from collections import Counter
from src.data_directories import *
from src.multi_ctrs.agents.agent import LLMAgent
from src.llm_setup.models import Gemini2Flash, Claude3Point7Sonnet, MistralLarge, Llama4Maverick17B, DeepSeekChatV3
from src.multi_ctrs.retrieval import Retrieval
from src.multi_ctrs.agents.helpers import *
from src.constants import CITIES
from src.data_directories import *
from src.constants import MISSING_CONTEXT_CONFIG_IDS
from src.k_base.context_retrieval import ContextRetrieval
from src.multi_ctrs.moderators.state_manager import StateManager
from src.multi_ctrs.moderators.helpers import post_process_response, weighted_reliability_score, min_max_normalize, \
    post_process_feedback, get_feedback_text, get_rejection_feedback, get_hallucination_feedback, post_process_response_llama, zscore_normalize, compute_success
import ast


class Moderator:
    def __init__(self, agents: List[LLMAgent], query_id: str, query: str, travel_filters: dict, k: int = 10,
                 max_retry: int = 5, city_score_init=False, relevance_weight=1, hallucination_weight=1,
                 reliability_weight=1, minimum_conversation_rounds=5):
        """
        Initializes a moderator that facilitates negotiation between two LLM agents.

        Parameters:
        - agents: list of LLMAgent
        - query_id: config_id
        - query: user's query
        - travel_filters: input for the multi-agent system
        - k: the number of recommended cities, default 10
        - max_retry: the number of max retry for initial feedback
        - city_score_init: set the initial city score as its relevance w.r.t. the filters if set True
        """
        self.agents = agents
        self.travel_filters = travel_filters
        self.query_id = query_id  # use to create folder while saving intermediate round results
        self.query = query
        self.retrieved_cities = self._retrieve_cities()

        self.relevance_weight = relevance_weight
        self.reliability_weight = reliability_weight
        self.hallucination_weight = hallucination_weight

        self.early_stopping = {
            0.2: None, 
            0.4: None, 
            0.6: None, 
            0.8: None
        }
        self.minimum_conversation_rounds = 5

        self.k = k
        self.max_retry = max_retry

        self.state_manager = StateManager()

        self.agent_reliability_scores = {agent.role: 1 for agent in self.agents}
        if city_score_init:
            self.city_scores = {city: self.get_city_relevance_score(city) for city in CITIES}
        else:
            self.city_scores = {city: 0 for city in CITIES}
        self.rejected_cities = set()
        self.collective_offer = []

    def _retrieve_cities(self):
        retrieval = ContextRetrieval()

        # Load configuration file
        print({"filters": self.travel_filters})
        results = retrieval.get_cities_from_config({"filters": self.travel_filters})

        return results

    def get_agent_reliability_score(self, agent, previous_offers, responses):
        """
        Computes agent reliability score
        """
        agent.reliability = weighted_reliability_score(
            k=self.k,
            prev_list=previous_offers[agent.role]["offers"],
            curr_list=responses[agent.role]["valid_offers"],
            prev_coll_offer=previous_offers["collective"]["offers"]
        )

    def get_city_relevance_score(self, city, filters=None):
        """
        Computes proportional relevance for city w.r.t filters
        """

        retriever = ContextRetrieval()

        if filters:
            agent_filters = {key: val for key, val in self.travel_filters.items() if key in filters}
            if len(agent_filters) < 1:
                agent_filters = {"seasonality": "low", "walkability": "great", "aqi": "great"}

            matched_filters = retriever.match_city_with_filters(city=city, filters=agent_filters)
            rel_score = len(matched_filters) / len(agent_filters)

        else:
            matched_filters = retriever.match_city_with_filters(city=city, filters=self.travel_filters)
            rel_score = len(matched_filters) / len(self.travel_filters)

        return rel_score
    
    def calculate_improvement(self, round):

        if round < self.minimum_conversation_rounds: 
            return 0

        round_0 = compute_success(
            cities=self.state_manager.rounds[0]['collective']['offers'],
            filters=self.travel_filters
        )
        
        round_score = compute_success(
                cities=self.collective_offer,
                filters=self.travel_filters,
        )

        if round_score == 1:
            return 1

        try:
            improvement = max(0, (round_score-round_0)/round_0)
        
        except ZeroDivisionError as e:
            improvement=0
            
        return improvement

    def get_agent_proportion_relevance_score(self, agent, responses, filter_based=False):
        """
        Computes agent relevance score w.r.t filter proportions
        """
        # valid_offers = responses[agent.role]['valid_offers']
        # tp = len([city for city in valid_offers if city in self.retrieved_cities])

        # agent.relevance = tp / len(self.retrieved_cities)

        valid_offers = responses[agent.role]['valid_offers']
        if filter_based:
            candidate_relevance = [self.get_city_relevance_score(city, filters=agent.specs['filters']) for city in
                                   valid_offers]

        else:
            candidate_relevance = [self.get_city_relevance_score(city) for city in valid_offers]

        agent.relevance = np.mean(candidate_relevance)

    def get_agent_relevance_score(self, agent, responses):
        """
        Computes agent relevance score
        """
        valid_offers = responses[agent.role]['valid_offers']
        tp = len([city for city in valid_offers if city in self.retrieved_cities])

        agent.relevance = tp / self.k

    def get_agent_hallucination_rate(self, agent, responses):
        """
        Computes agent hallucination rate - the higher the hit rate, the lower should be the impact of hallucination rate, but the city scores should show a decrease if an agent has hallucinated
        """

        hit_rate = len(set(responses[agent.role]['candidates']) & self.city_scores.keys()) / len(
            set(responses[agent.role]['candidates']))
        agent.hallucination = - (1 - hit_rate)  # this is so that the hallucination DECREASES the final score.

    def identify_hallucinations_and_rejections(self, response):
        """
        Identifies those cities which have hallucinated or have already been rejected, but have been recommended by the agent.
        """
        hallucinated = [city for city in response if city not in CITIES]
        rejected = [city for city in response if (city in self.rejected_cities)]

        return hallucinated, rejected

    def initiate_feedback(self, candidates, agent):
        """
        Function that initiates the feedback loop in the case of hallucinations/rejected cities in the recommended list. 
        """
        # TODO : does hallucination rate consider the rate after end of initial feedback or every iteration - end of initial
        hallucinated, rejected = self.identify_hallucinations_and_rejections(candidates)
        print(hallucinated, rejected)
        valid_candidates = [candidate for candidate in candidates if
                            candidate not in hallucinated and candidate not in rejected]

        if valid_candidates == candidates:
            return [valid_candidates, candidates]

        print(f"initiate feedback for agent {agent.role}")

        args_dict = {
            "travel_filters": self.travel_filters,
            "query": self.query,
            "travel_filters": self.travel_filters,
            "query": self.query,
            "mod_output": {
                "query": self.query,
                "k_invalid": len(hallucinated) + len(rejected),
                "travel_filters": self.travel_filters,
                "valid_candidates": valid_candidates,
                "invalid_candidates": hallucinated + rejected,
                "available_candidates": self.city_scores.keys()
            }
        }
        response = agent.run(
            round=-1,
            args=args_dict
        )

        new_candidates = post_process_feedback(response, candidates)

        # check if model still hallucinated
        hallucinated, rejected = self.identify_hallucinations_and_rejections(new_candidates)
        new_valid_candidates = [candidate for candidate in new_candidates if
                                candidate not in hallucinated and candidate not in rejected]

        return [new_valid_candidates, new_candidates]

    def compute_rejections(self, round, responses):
        """
        Computes the new rejected cities - a city is defined as rejected if and only if (i) it was present in the current offer and (ii) both agents exclude it from their candidates

        """
        if round > 0:
            rejects = []
            for agent in self.agents:
                rejected = set(self.collective_offer) - set(responses[agent.role]['valid_offers'])
                responses[agent.role]['rejections'] = list(rejected)
                rejects.append(rejected)

            coll_rejection = set(rejects[0])
            for lst in rejects[1:]:
                coll_rejection = coll_rejection.intersection(lst)

            self.rejected_cities.update(coll_rejection)
        else:
            for agent in self.agents:
                responses[agent.role]['rejections'] = None

        return responses

    def compute_majority_rejections(self, round, responses):
        """
        Computes the new rejected cities - a city is defined as rejected if and only if (i) it was present in the current offer and (ii) minimum two agents exclude it from their candidates

        """
        if round > 0:
            rejects = []
            for agent in self.agents:
                rejected = set(self.collective_offer) - set(responses[agent.role]['valid_offers'])
                responses[agent.role]['rejections'] = list(rejected)
                rejects.append(rejected)
            counter = Counter()
            for lst in rejects:
                counter.update(set(lst))

            coll_rejection = [item for item, count in counter.items() if count >= 2]

            self.rejected_cities.update(coll_rejection)
        else:
            for agent in self.agents:
                responses[agent.role]['rejections'] = None

        return responses
    
    def compute_aggresive_rejections(self, round, responses):
        """
        Computes the new rejected cities - a city is defined as rejected if and only if (i) it was present in the current offer and (ii) any agent exclude it from their candidates

        """
        if round > 0:
            coll_rejection = []
            for agent in self.agents:
                rejected = set(self.collective_offer) - set(responses[agent.role]['valid_offers'])
                responses[agent.role]['rejections'] = list(rejected)
                coll_rejection.extend(responses[agent.role]['rejections'][:agent.k_reject])

            coll_rejection = list(set(coll_rejection))

            self.rejected_cities.update(coll_rejection) 
        else:
            for agent in self.agents:
                responses[agent.role]['rejections'] = None
        
        return responses
        
    def update_state(self, round, responses):
        """
        Update the state manager of the moderator
        """

        state_manager_dict = {}
        improvement = self.calculate_improvement(round)
        print(f"Improvement: {improvement}")

        for threshold, val in self.early_stopping.items():
            if round > 0 and improvement > threshold and val is None: 
                print(f"Percentage Improvement of Moderator Success at round {round} > Early Stopping Threshold: {threshold}!")
                self.early_stopping[threshold] = round

        print(f"Early Stopping Thresholds at the end of round {round}: {self.early_stopping}")

        for agent in self.agents:
            state_manager_dict[agent.role] = {
                "relevance_score": agent.relevance,
                "reliability_score": agent.reliability,
                "hallucination_rate": agent.hallucination,
                "offers": responses[agent.role]['valid_offers'],
                "rejections": responses[agent.role]['rejections'],
                "city_scores": None, 
                "early_stopping": None
            }

        state_manager_dict['collective'] = {
            "relevance_score": None,
            "reliability_score": None,
            "hallucination_rate": None,
            "offers": self.collective_offer,
            "rejections": list(self.rejected_cities) if self.rejected_cities else None,
            "city_scores": self.city_scores.copy(), 
            "early_stopping": self.early_stopping
        }
        self.state_manager.insert_round_state(
            round_number=round,
            round_dict=state_manager_dict
        )

    def generate_feedback(self, responses):
        """
        Generates the feedback provided to the agent
        """

        for agent in self.agents:
            offers = responses[agent.role]["valid_offers"]
            proportion = len(set(offers) & set(self.collective_offer)) / self.k
            agent.feedback = get_feedback_text(proportion, self.k)

    def generate_feedback_with_rejection(self, responses):
        """
        Generates the feedback provided to the agent
        """

        for agent in self.agents:
            offers = responses[agent.role]["valid_offers"]
            proportion = len(set(offers) & set(self.collective_offer)) / self.k
            agent.feedback = get_feedback_text(proportion, self.k)

            agent.feedback += get_hallucination_feedback(agent.hallucination, self.k)

            rejections = responses[agent.role]["rejections"]
            if rejections and (len(rejections) > agent.k_reject):
                agent.feedback += get_rejection_feedback(len(rejections), agent.k_reject)

    def negotiate(self, round, previous_offers, relevance_method=None, rejection_method=None, feedback_method=None, smoothing_weight=None, standardize=False, agent_assessment=True, with_examples=True):
        """
        Main negotiation function, sequence of actions: 

        1. Initial feedback round : check for rejected cities/hallucinations; if exist, asks to regenerate
        2. Aggregating collective rejection : comparing between new candidates with collective offer
        3. Compute agent reliability given the rank deviations from previous round
        4. Compute city_agent_scores and relevant_scores for each city
        5. Compute final score for each city
        6. Compute the new offer (ranked top k)
        7. Provide the new offer, rejected candidates and other options to each agent

        Note: relevance_method can either be "proportion", "hit_rate" or "prop_filter_based"
        
        """
        # collect valid offers
        responses = self.collect_response(round, examples=with_examples)

        # compute new rejections 
        if rejection_method == "majority":
            responses = self.compute_majority_rejections(round, responses)
        elif rejection_method == "aggressive":
            responses = self.compute_aggresive_rejections(round, responses)
        else:
            responses = self.compute_rejections(round, responses)

        # remove rejected cities from scoring
        for city in self.rejected_cities:
            if city in self.city_scores:
                del self.city_scores[city]

        # compute the evaluation score
        for agent in self.agents:

            offers = responses[agent.role]['valid_offers']

            # compute and update reliability score
            if round > 0:
                self.get_agent_reliability_score(
                    agent=agent,
                    previous_offers=previous_offers,
                    responses=responses
                )

            # compute and update relevance score
            if relevance_method == "proportion":
                self.get_agent_proportion_relevance_score(
                    agent=agent,
                    responses=responses
                )
            elif relevance_method == "prop_filter_based":
                self.get_agent_proportion_relevance_score(
                    agent=agent,
                    responses=responses,
                    filter_based=True
                )
            else:
                self.get_agent_relevance_score(
                    agent=agent,
                    responses=responses
                )

            # compute and update hallucination rate
            self.get_agent_hallucination_rate(
                agent=agent,
                responses=responses
            )

            # compute new evaluation scores
            for city in [offer for offer in offers if offer not in self.rejected_cities]:
                city_rank = offers.index(city) + 1
                if agent_assessment:
                    if smoothing_weight:
                        self.city_scores[city] = (1 - smoothing_weight) * self.city_scores[city] + smoothing_weight * (
                                    1 / city_rank) * (self.hallucination_weight * agent.hallucination + self.relevance_weight * agent.relevance + self.reliability_weight * agent.reliability)
                    else:
                        self.city_scores[city] += (1 / city_rank) * (
                                    agent.hallucination + agent.relevance + agent.reliability)
                else:
                    if smoothing_weight:
                        self.city_scores[city] = (1 - smoothing_weight) * self.city_scores[city] + smoothing_weight * (1 / city_rank)
                    else:
                        self.city_scores[city] += (1 / city_rank)



        if standardize: 
            self.city_scores = zscore_normalize(self.city_scores)
        else:
            self.city_scores = min_max_normalize(self.city_scores)
        new_city_scores = dict(sorted(self.city_scores.items(), key=lambda item: item[1], reverse=True))

        # update collective offer
        self.collective_offer = [k for k, v in new_city_scores.items()][:self.k]
        if feedback_method=="rejection":
            self.generate_feedback_with_rejection(responses)
        else:
            self.generate_feedback(responses)
        self.update_state(round, responses)
        return responses

    def build_args_dict(self, round, agent):

        if round < 1:
            return {
                "travel_filters": self.travel_filters,
                "query": self.query
            }
        else:
            return {
                "mod_output": {
                    "query": self.query,
                    "travel_filters": self.travel_filters,
                    "prev_output": self.state_manager.get_offers_by_item(round - 1, agent.role),
                    "curr_offer": self.collective_offer,
                    "candidate_cities": [k for k in self.city_scores.keys()],
                    "agent_feedback": agent.feedback
                }}

    def collect_response(self, round, examples=True):
        responses = {}
        for agent in self.agents:
            args_dict = self.build_args_dict(round, agent)
            agent_response = agent.run(round, args_dict,examples=examples)
            if agent.model == Llama4Maverick17B:
                candidates = post_process_response_llama(agent_response)
            else:
                candidates = post_process_response(agent_response)
            print(candidates)
            valid_offers, new_candidates = self.initiate_feedback(candidates, agent)
            response = {"full_response": agent_response,
                        "candidates": new_candidates,
                        "valid_offers": valid_offers}
            responses[agent.role] = response

        return responses


def run():
    # define experiment settings
    k_offer = 10
    k_reject = 3
    max_retry = 1
    rounds = 4
    query_id = "c_p_27_pop_low_medium"
    query = "European cities with low popularity to visit in February with cultural attractions and live music venues."
    travel_filters = {
        "popularity": "low",
        "month": "February",
        "interests": "Arts & Entertainment"
    }

    models = {
        "popularity": Llama4Maverick17B,
        "constraint": Claude3Point7Sonnet,
        "interest": DeepSeekChatV3
    }

    # travel_filters = {'popularity': 'medium', 'interests': 'Food', 'budget': 'low', 'walkability': 'great'}

    # create and define agents
    agents = []
    for role, model in models.items():
        agents.append(LLMAgent(model, role, k_offer, k_reject))

    # initialize moderator
    mod = Moderator(agents, query_id, query, travel_filters, k_offer, max_retry)

    for round in range(rounds):
        # load prev round scores
        if round > 0:
            previous_offers = mod.state_manager.read_round(round - 1)
        else:
            previous_offers = []

        mod.negotiate(round, previous_offers)

    mod.state_manager.export_state_to_json(f"./test_{query_id}")


if __name__ == "__main__":
    run()
