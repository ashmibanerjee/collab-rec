import json
from typing import List, Dict, Any, Callable
from src.data_directories import *
from src.llm_setup.models import Gemini2Flash, Gemini1Point5Pro, Claude3Point5Sonnet, Llama3Point2Vision90B, GPT4, \
    GPT4o, GPTo1Mini, Claude3Point7Sonnet, MistralLarge, Gemini2Point5Flash
from src.multi_ctrs.retrieval import Retrieval
from src.multi_ctrs.agents.helpers import *
from src.constants import CITIES, MODEL_LOCATIONS, ROLES, MISSING_CONTEXT_CONFIG_IDS
import argparse
import logging
import time
from jinja2 import Environment, FileSystemLoader
from src.data_directories import *
from src.llm_setup.models import Gemini2Flash, Gemini1Point5Pro, Claude3Point5Sonnet, Llama3Point2Vision90B
import random

logger = logging.getLogger(__name__)

SYS_TEMPLATE = "general_prompt_sys.txt"
USR_TEMPLATE = "general_prompt_user.txt"

# set the random seed
random.seed(42)


def _get_specifications(role):
    with open(f"{prompts_dir}agents/agent_specifications.json", "r") as fp:
        specs = json.load(fp)

    return specs[role]


def _get_model_location(model):
    if model == Claude3Point7Sonnet:
        return ["us-east5", "europe-west1"]
    elif model == MistralLarge:
        return ["europe-west4", "us-central1"]
    elif model == Gemini2Point5Flash:
        return ["us-central1"]
    else:
        return MODEL_LOCATIONS


class LLMAgent:
    """
    Template class for an agent 
    
    """

    def __init__(self, model, role, k_offer, k_reject=None):
        """
        Instantiates agent with model and role; the role has to belong to one of the predefined roles mentioned in ROLES

        """
        # TODO : check if role in ROLES
        self.model = model
        self.model_location = _get_model_location(model)
        self.role = role
        self.specs = _get_specifications(role)

        self.k_reject = k_reject
        self.k_offer = k_offer
        self.specs.update({"k_reject": self.k_reject,
                           "k_offer": self.k_offer})

        # initial relevance, reliability and hallucination values - set at max for each because we initially assume
        # an agent is fully relevant, fully reliable and does not hallucinate
        self.relevance = 1
        self.reliability = 1
        self.hallucination = 0
        self.feedback = ""

    def _build_prompt_single_agent(self, context):
        """
        Prompt for the single agent setting
        """

        context.update({"k_offer": self.k_offer})

        sys_prompt = render_prompt(
            round="single",
            role='system',
            template=SYS_TEMPLATE,
            specs=context
        )

        usr_prompt = render_prompt(
            round="single",
            role='user',
            template=USR_TEMPLATE,
            specs=context
        )

        return [sys_prompt, usr_prompt]

    def _build_prompt_initialize(self, context):
        """
        Initialization prompt for round 1 (prior to negotiation)
        Params: 
        - context: dict of relevant filters from synthTRIPS 
        """
        travel_filters = {k: v for k, v in context.items() if k in self.specs["filters"]}
        final_context = self.specs.copy()
        final_context.update({"filters": travel_filters, "cities": CITIES})

        sys_prompt = render_prompt(
            round="init",
            role='system',
            template=SYS_TEMPLATE,
            specs=final_context
        )

        usr_prompt = render_prompt(
            round="init",
            role='user',
            template=USR_TEMPLATE,
            specs=final_context
        )

        return [sys_prompt, usr_prompt]

    def _build_prompt_iterate(self, mod_output, examples=True):
        """
        Iteration prompt for rounds 2 onwards. 

        Params: - mod_output: moderator's output: this is a dictionary consisting of the offer, previous agent
        output, other candidates and rejected cities
        """
        context = self.specs.copy()
        context.update(mod_output)
        context['filters'] = {k: v for k, v in context["travel_filters"].items() if k in self.specs["filters"]}

        sys_prompt = render_prompt(
            round="iter",
            role='system',
            template=SYS_TEMPLATE,
            specs=context
        )

        usr_prompt = render_prompt(
            round="iter",
            role='user',
            template=USR_TEMPLATE,
            specs=context
        )

        # append usr prompt
        if examples:
            dir = f"{prompts_dir}agents/examples/{self.role}.txt"
            with open(dir, "r") as fp:
                usr_prompt["content"] += fp.read()

        print(usr_prompt)

        return [sys_prompt, usr_prompt]

    def _build_prompt_regenerate(self, mod_output):
        # TODO : should figure out previous output mechanism
        """
        Prompt for initial feedback

        Params: 
        - mod_output: moderator's output: this is a dictionary consisting of the offer, previous agent output, other candidates and rejected cities
        """
        context = self.specs.copy()
        context.update(mod_output)
        context['filters'] = {k: v for k, v in context["travel_filters"].items() if k in self.specs["filters"]}

        sys_prompt = render_prompt(
            round="feedback",
            role='system',
            template=SYS_TEMPLATE,
            specs=context
        )

        usr_prompt = render_prompt(
            round="feedback",
            role='user',
            template=USR_TEMPLATE,
            specs=context
        )
        print(usr_prompt)
        return [sys_prompt, usr_prompt]

    def _build_prompt(self, round, args, examples=True):
        """
        Driver function to initialize prompts. Default experiment setting is MAMI
        """

        if round == 0:
            prompt = self._build_prompt_initialize(
                context=args["travel_filters"]
            )
        elif round > 0:
            prompt = self._build_prompt_iterate(
                mod_output=args["mod_output"],
                examples=examples
            )

        else:
            prompt = self._build_prompt_regenerate(
                mod_output=args["mod_output"]
            )

        return prompt

    def _generate(self, prompt):
        """Generate a response using the specified model locations."""

        for model_location in self.model_location:
            try:
                llm = self.model(location=model_location)
                response = llm.generate(messages=prompt)
                return response
            except Exception as e:
                logger.error(f"Error with model at {model_location}: {e}")
                time.sleep(10)
                continue

        logger.error("All model locations have been tried and failed.")
        return None

    def run(self, round, args, examples=True, exp="mami"):
        """
        Driver function to build and run the prompt. 
        
        Params: 
        - Round: to indicate whether prompt is required for initialization or negotiation
        - Args: Dictionary of arguments containing filters + mod_output
        - Exp: experiment setting - SASI, MAMI, MASI; to know what prompt needs to be rendered
        """

        if "is_baseline" in exp:
            prompt = self._build_prompt_single_agent(context=args)

        else:
            if round > 0 and "mod_output" not in args:
                print("Error! Missing moderator output for negotiation.")
                return None
            # elif "travel_filters" not in args: 
            #     print("Error! Missing travel filters for negotiation.")
            #     return None

            prompt = self._build_prompt(round, args, examples=examples)

        print(f"Running round {round} for role {self.role}")
        time.sleep(2)

        return self._generate(prompt)


def test():
    rej_div = ["London", "Barcelona", "Amsterdam", "Krakow"]
    rej_pref = ["Milan", "Ljubljana", "Zagreb", "Bratislava"]
    rej_all = rej_div + rej_pref
    args_dict = {
        "travel_filters": {
            "popularity": "High",
            "budget": "Low",
            "interests": "Outdoors & Recreation"}
    }

    args_dict_div = {
        "travel_filters": {
            "popularity": "High",
            "budget": "Low",
            "interests": "Outdoors & Recreation"},
        "mod_output": {
            "prev_output": ["Paris", "Rome", "Vienna", "Madrid", "Milan", "Prague", "Istanbul", "Moscow", "Berlin",
                            "Budapest"],
            "curr_offer": ["Paris", "Rome", "Madrid", "Vienna", "Budapest", "Prague", "Berlin", "Istanbul", "Munich",
                           "London"],
            "candidate_cities": [city for city in CITIES if city not in rej_all],
            "rejected_cities": rej_div
        }
    }

    args_dict_pref = {
        "travel_filters": {
            "popularity": "High",
            "budget": "Low",
            "interests": "Outdoors & Recreation"},
        "mod_output": {
            "prev_output": ["Paris", "Budapest", "Krakow", "Rome", "Madrid", "Vienna", "Prague", "Berlin", "Munich",
                            "London"],
            "curr_offer": ["Paris", "Rome", "Madrid", "Vienna", "Budapest", "Prague", "Berlin", "Istanbul", "Munich",
                           "London"],
            "candidate_cities": [city for city in CITIES if city not in rej_all],
            "rejected_cities": rej_pref
        }
    }

    # initialize agents
    k_reject = 2
    k_offer = 10

    div_agent = LLMAgent(Gemini1Point5Pro, "popularity", k_offer, k_reject)
    pref_agent = LLMAgent(Gemini1Point5Pro, "constraint", k_offer, k_reject)

    initial_candidates_div = div_agent.run(1, args_dict_div)
    initial_candidates_pref = pref_agent.run(1, args_dict_pref)

    print(initial_candidates_div, initial_candidates_pref)


if __name__ == "__main__":
    test()
