from typing import Dict, Any, Set, List
from collections import Counter
import logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------


def remove_rejected_from_scores(
        city_scores: Dict[str, float],
        rejected_cities_globally: Set[str]
) -> Dict[str, float]:
    for city in rejected_cities_globally:
        city_scores.pop(city, None)

    return city_scores


# ------------------------------------------------------------------
# Voting strategies
# ------------------------------------------------------------------

def _compute_aggressive_rejections(
        rejects: List[Set[str]],
        parsed_responses: Dict[str, Dict[str, Any]]
) -> Set[str]:
    aggressive = set()

    for (agent, data), rejected_set in zip(parsed_responses.items(), rejects):
        k_reject = data.get("k_reject", len(rejected_set))
        aggressive.update(list(rejected_set)[:k_reject])

    return aggressive


def _compute_majority_rejections(rejects: List[Set[str]]) -> Set[str]:
    if not rejects:
        return set()

    counter = Counter()
    for r in rejects:
        counter.update(r)

    num_agents = len(rejects)
    threshold = (num_agents // 2) + 1  # ⌈|A| / 2⌉

    return {
        city for city, count in counter.items()
        if count >= threshold
    }


def _compute_unanimous_rejections(rejects: List[Set[str]]) -> Set[str]:
    if not rejects or len(rejects) == 1:
        return set()

    intersection = rejects[0].copy()
    for r in rejects[1:]:
        intersection &= r
    return intersection


class RejectionManager:
    """Manages city rejections across agents."""

    def __init__(self, rejection_method: str = "majority"):
        self.rejection_method = rejection_method

    def compute_rejections(
            self,
            parsed_responses: Dict[str, Dict[str, Any]],
            collective_offer: List[str],
            rejected_cities_globally: Set[str],
            round_number: int
    ) -> Set[str]:

        # ---- Round 1: no rejection ----
        if round_number == 1 or not collective_offer:
            for data in parsed_responses.values():
                data["rejections"] = []
            return rejected_cities_globally

        rejects: List[Set[str]] = []

        # ---- Step 1: per-agent omissions ----
        for agent, data in parsed_responses.items():
            # IMPORTANT: use valid_offers (paper-faithful)
            agent_offer = set(
                data.get("valid_offers", data.get("candidates", []))
            )

            rejected_by_agent = set(collective_offer) - agent_offer
            data["rejections"] = list(rejected_by_agent)
            rejects.append(rejected_by_agent)

            logger.debug(
                f"{agent}: omitted {rejected_by_agent} from previous collective offer"
            )

        # ---- Step 2: voting ----
        if self.rejection_method == "unanimous":
            newly_rejected = _compute_unanimous_rejections(rejects)

        elif self.rejection_method == "aggressive":
            newly_rejected = _compute_aggressive_rejections(rejects, parsed_responses)

        else:  # majority (default)
            newly_rejected = _compute_majority_rejections(rejects)

        # ---- Step 3: cumulative aggregation ----
        if newly_rejected:
            rejected_cities_globally.update(newly_rejected)
            logger.info(
                f"Round {round_number}: newly rejected cities = {sorted(newly_rejected)}"
            )

        return rejected_cities_globally
