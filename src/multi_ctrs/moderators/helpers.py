import re, ast
import numpy as np 
from num2words import num2words
from src.k_base.context_retrieval import ContextRetrieval

def post_process_response(response:str):
    pat = r"\[([^\[\]]*?(?:['\"][^'\"]+['\"],?\s*)+?)\]"
    response = response.replace("\n", "")
    match = re.findall(pat, response, re.DOTALL)
    print("response:", response)
    # print("matches:", match[-1].strip())
    response_list = None
    for i in range(len(match), 0, -1):
        if len(list(ast.literal_eval(match[i-1].strip()))) == 10:
            response_list = list(ast.literal_eval(match[i-1].strip()))
            break
    return response_list

def post_process_response_llama(response:str):
    pat = r"\[([^\[\]]*?(?:['\"][^'\"]+['\"],?\s*)+?)\]"
    full_response = response.replace("\n", "")
    match = re.findall(pat, full_response)
    print(full_response)
    try: 
        return list(ast.literal_eval(match[-1]))
    except:
        print(full_response)
        raise


def post_process_feedback(response: str, candidates):
    response = response.replace("\n", "")
    print(response)

    response = re.sub(r'^json\s*', '', response,  flags=re.IGNORECASE)

    # Match the full JSON object
    pat = r'\{[\s\S]*\}'  # Greedy, includes newlines and all content
    match = re.findall(pat, response, re.DOTALL)

    if not match:
        print("No JSON object found in response.")
        return candidates

    try:
        replacements = ast.literal_eval(match[0])
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return candidates

    for key, value in replacements.items():
        try: 
            index = candidates.index(key)
            candidates[index] = value
        except ValueError: 
            print(f"City {key} from feedback not found in list of candidates!") 
        
    return candidates
    

def weighted_reliability_score(k, prev_list, curr_list, prev_coll_offer):
        """
        Computes the weighted reliability score
        """
        base_drop_penalty = k
        base_new_penalty = k

        score = 0

        prev_set = set(prev_list)
        curr_set = set(curr_list)

        for city in prev_set & curr_set:
            old_rank = prev_list.index(city)
            new_rank = curr_list.index(city)
            # weight = 1 / (old_rank + 1)  # rank 1 has weight 1.0, rank 10 has 0.1
            score += abs(new_rank - old_rank)

        for city in prev_set - curr_set:
            old_rank = prev_list.index(city)
            # weight = 1 / (old_rank + 1)
            score += base_drop_penalty

        for city in curr_set - prev_set:
            if city in prev_coll_offer:
                offer_rank = prev_coll_offer.index(city)
            else:
                offer_rank = np.inf
            new_rank = curr_list.index(city)
            # weight = 1 / (new_rank + 1)
            score += min(base_new_penalty, abs(new_rank - offer_rank))

        worst_case_score = len(prev_list) * (base_drop_penalty + base_new_penalty)
        reliability = 1 - (score / worst_case_score)
        return max(reliability, 0)

def min_max_normalize(data):
    """
    Normalizes the city scores
    """

    values = np.array(list(data.values()))
    min_val = np.min(values)
    max_val = np.max(values)

    # Normalize
    normalized_data = {k: (v - min_val) / (max_val - min_val) for k, v in data.items()}
    
    return normalized_data

def zscore_normalize(data):
    """
    Standardizes the city scores using mean and standard deviation.
    """
    values = np.array(list(data.values()))
    mean_val = np.mean(values)
    std_dev = np.std(values)  # Population std; use ddof=1 for sample std

    # Standardize
    standardized_data = {k: (v - mean_val) / std_dev for k, v in data.items()}
    
    return standardized_data


def get_feedback_text(proportion, k):
    """
    Returns feedback given the ratio of common cities
    """

    if proportion > 0.5: 
        return f"Congrats! More than half of your previous recommendations were relevant. Please continue to maintain this in your next recommendation list."
    elif proportion >= 0.3 and proportion <= 0.5: 
        return f"You have offered {num2words(proportion * k)} relevant cities, but you could look at offering more relevant cities based on your previous offer and user requirements."
    elif proportion >= 0.1 and proportion < 0.3: 
        return f"Very few (only {num2words(proportion * k)}) of your cities were considered good enough to be included in the Current Offer as they were not relevant. Please recommend more relevant cities."
    else:
        return f"None of your recommended cities made it to the Current Offer!! This means that none of them were relevant. Please stick to the instructions provided to you and provide better recommendations."
    
def get_hallucination_feedback(hallucination_rate, k):
    if hallucination_rate > 0:
        return f"You offered {num2words(hallucination_rate * k)} cities that are already rejected. This violates the instruction and the user will reject your recommendation."
    else:
        return ""

def get_rejection_feedback(rejected_cities, k_reject):
    """
    Returns feedback when the agent is too aggresive
    """    
    return f"Warning! You rejected {num2words(rejected_cities)} cities in the previous round. This violates the instruction and the user will reject your recommendation. Consider the cities in the collective offer more carefully, you can only replace up to {num2words(k_reject)} cities."

def compute_success(cities, filters, sasi=False):
    """
    Computes proportional relevance for city w.r.t filters
    """

    retriever = ContextRetrieval()
    
    for city in cities:
        matched_filters = retriever.match_city_with_filters(city=city, filters=filters)
        unmatched = [key for key in filters.keys() if key not in matched_filters.keys()]
        rel_score = len(matched_filters)/len(filters)
    
    return rel_score