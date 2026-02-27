from typing import Dict, List, Any
from num2words import num2words

from src.k_base.context_retrieval import ContextRetrieval


def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return scores
    vals = list(scores.values())
    min_v, max_v = min(vals), max(vals)
    if min_v == max_v:
        return {k: 0.0 for k in scores}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}


def zscore_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return scores
    vals = list(scores.values())
    mean_val = sum(vals) / len(vals)
    variance = sum((x - mean_val) ** 2 for x in vals) / len(vals)
    std_dev = variance ** 0.5

    if std_dev == 0:
        return {k: 0.0 for k in scores}

    return {k: (v - mean_val) / std_dev for k, v in scores.items()}


def get_feedback_text(proportion, k):
    """
    Returns feedback given the ratio of common cities.
    Emphasizes the constraint of keeping at least 7 cities from collective offer.
    """
    min_keep = 7
    max_replace = 3

    if proportion >= 0.7:  # Kept 7+ cities
        return (f"Good! You kept {num2words(int(proportion * k))} cities from the Current Collective Offer. "
                f"Remember: you MUST keep at least {min_keep} cities from the Collective Offer and can replace at most {max_replace}. "
                f"Please continue to respect this constraint in your next recommendation list.")
    elif proportion >= 0.5 and proportion < 0.7:
        return (f"Warning! You only kept {num2words(int(proportion * k))} cities from the Current Collective Offer. "
                f"You MUST keep at least {min_keep} cities from the Collective Offer (you can replace at most {max_replace}). "
                f"Violating this constraint means your recommendations will be rejected.")
    else:
        return (f"CONSTRAINT VIOLATED! You kept only {num2words(int(proportion * k))} cities from the Current Collective Offer. "
                f"You MUST keep at least {min_keep} cities and can replace at most {max_replace}. "
                f"Your recommendations must come from: (1) Current Collective Offer (keep ≥7), (2) Additional Candidates provided (use ≤3 for replacement). "
                f"Do NOT recommend cities outside these sets!")


def get_hallucination_feedback(hallucination_rate, k):
    """
    Returns feedback about hallucinated or rejected cities.
    """
    if hallucination_rate > 0:
        return (f"\n\nCRITICAL: You recommended {num2words(int(hallucination_rate * k))} cities that are either "
                f"(1) not in the catalog, or (2) already in the rejected list. "
                f"You MUST ONLY choose from: Current Collective Offer + Additional Candidates provided. "
                f"Do NOT recommend rejected cities!")
    else:
        return ""


def get_rejection_feedback(num_rejected: int, k_reject: int = 3) -> str:
    """
    Returns feedback about rejections (replacing too many cities from collective offer).
    """
    if num_rejected > k_reject:
        return (f"\n\nWarning! You replaced {num2words(num_rejected)} cities from the Collective Offer in the previous round. "
                f"You can replace at most {num2words(k_reject)} cities. "
                f"This means you must keep at least {num2words(10 - k_reject)} cities from the Current Collective Offer. "
                f"Consider the cities in the Collective Offer more carefully.")
    return ""
