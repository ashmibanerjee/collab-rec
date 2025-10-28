import pandas as pd
from src.multi_ctrs.retrieval import CSVRetrieval
from src.multi_ctrs.agent import Agent
from src.data_directories import ctrs_kb_dir



class Agent:
    def __init__(self, llm_model: Callable[[str], str], retrieval: Retrieval):
        """
        Initializes the RAG agent with a given LLM model and a retrieval component.
        :param llm_model: A callable LLM model that takes a prompt and returns a response.
        :param retrieval: A retrieval component for fetching relevant documents.
        """
        self.llm_model = llm_model
        self.retrieval = retrieval

    def retrieve_relevant_data(self, query_input: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves relevant documents from the retrieval component to augment the negotiation process.
        :param query_input: The initial user query.
        :param top_k: The number of relevant documents to retrieve.
        :return: A list of relevant retrieved documents.
        """
        return self.retrieval.retrieve_candidates(query_input, top_k)

    def augment_data(self, query_input: str, current_offer: Dict[str, Any], 
                     top_candidates: List[Dict[str, Any]], rejected_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Augments negotiation data with relevant retrieved information.
        :param query_input: The initial user query.
        :param current_offer: The offer currently being negotiated.
        :param top_candidates: The top available candidates.
        :param rejected_list: The previously rejected offers.
        :return: A list containing all available information, including retrieved data.
        """
        retrieved_docs = self.retrieve_relevant_data(query_input)
        return top_candidates + retrieved_docs

    def build_prompt(self, query_input: str, current_offer: Dict[str, Any], 
                     top_candidates: List[Dict[str, Any]], rejected_list: List[Dict[str, Any]]) -> str:
        """
        Constructs a structured prompt with system and user role messages for any LLM to facilitate negotiation.
        :param query_input: The initial user query.
        :param current_offer: The offer currently being negotiated.
        :param top_candidates: The top available candidates (e.g., alternative options).
        :param rejected_list: The previously rejected offers.
        :return: A formatted prompt string.
        """
        augmented_candidates = self.augment_data(query_input, current_offer, top_candidates, rejected_list)
        
        system_message = (
            "System: You are an AI-powered negotiation assistant. Your task is to help the user find the best possible offer "
            "by analyzing the current offer, available alternatives, previously rejected offers, and retrieved knowledge. Your goal is to "
            "maximize user satisfaction and recommend the best next action.\n"
        )
        
        user_message = (
            f"User: I am looking for an offer based on this query: '{query_input}'.\n"
            f"Current Offer:\n{json.dumps(current_offer, indent=2)}\n"
            "Available Candidates:\n" + "\n".join(json.dumps(c, indent=2) for c in augmented_candidates) + "\n"
            "Previously Rejected Offers:\n" + "\n".join(json.dumps(r, indent=2) for r in rejected_list) + "\n"
            "Please provide the best negotiation strategy or alternative recommendation."
        )
        
        return system_message + user_message

    def negotiate(self, query_input: str, current_offer: Dict[str, Any], 
                  top_candidates: List[Dict[str, Any]], rejected_list: List[Dict[str, Any]]) -> str:
        """
        Runs the negotiation process using the LLM with retrieved and augmented data.
        :param query_input: The initial user query.
        :param current_offer: The offer currently being negotiated.
        :param top_candidates: The top available candidates (alternative options).
        :param rejected_list: The previously rejected offers.
        :return: The LLM-generated negotiation response.
        """
        prompt = self.build_prompt(query_input, current_offer, top_candidates, rejected_list)
        response = self.llm_model(prompt)
        return response  
class DiversityRetrieval(CSVRetrieval):
    def __init__(self, csv_file: str=None):
        if not csv_file:
            csv_file = ctrs_kb_dir + "diversity_agent_kb.csv"
        super().__init__(csv_file)

    def retrieve(self, filter: dict, top_k: int, available_candidates: list):
        available_data = self.data[self.data["city"].isin(available_candidates)]
        # first retrieve only filtered cities
        filtered_data = self.filter_by_columns(available_data, filter)
        top_candidates = self.rank_candidates(filtered_data, "normalized_poi", top_k, ascending=True)
        # if top candidates not enough, add the rest
        top_candidates.extend(self.rank_candidates(available_data[~available_data["city"].isin(filtered_data["city"].unique())],
                                                   "normalized_poi", top_k-len(top_candidates), ascending=True))
        
        return top_candidates

class DiversityAgent(Agent):
    def __init__(self, llm_model):
        super().__init__(llm_model, DiversityRetrieval())


def test():
    ret = DiversityRetrieval()
