"""
Scoring Manager for computing agent metrics (reliability, relevance, hallucination).
"""
import logging
from pathlib import Path
from datetime import datetime

from adk.agents.moderator_utils.compute_scores import compute_hallucination_rate, \
    compute_agent_reliability, update_city_scores, compute_agent_success_scores

logger = logging.getLogger(__name__)


def setup_scoring_logger(log_dir: str = "logs") -> logging.Logger:
    """
    Set up a separate file handler for scoring logs.

    Args:
        log_dir: Directory to store log files

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"scoring_metrics_{timestamp}.log"

    # Get logger
    scoring_logger = logging.getLogger(__name__)

    # Check if file handler already exists to avoid duplicates
    has_file_handler = any(isinstance(h, logging.FileHandler) for h in scoring_logger.handlers)

    if not has_file_handler:
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        scoring_logger.addHandler(file_handler)
        scoring_logger.setLevel(logging.DEBUG)

        scoring_logger.info(f"Scoring logs will be written to: {log_file}")

    return scoring_logger


# Map agent names to their assigned filter keys
AGENT_FILTER_MAPPING = {
    'personalization_agent': {"budget", "month", "interests"},
    'sustainability_agent': {"aqi", "walkability", "seasonality"},
    'popularity_agent': {"popularity"}
}

from constants import CITIES


class ScoringManager:
    """
    Paper-faithful scoring manager for Collab-REC.
    Implements Equations (4), (5), (6), and (11) exactly.
    """

    def __init__(self, k: int, retriever, ablated_component: str | None = None):
        """
        Args:
            k: recommendation list length
            catalog: external KB catalog C
            retriever: KB interface used for relevance
            ablated_component: if not None, the name of the component to ablate (e.g., hallucination, reliability, success or rank)
        """
        self.k = k
        self.catalog = CITIES
        self.retriever = retriever
        self.ablated_component = ablated_component

    # ------------------------------------------------------------------
    # Agent-level metrics
    # ------------------------------------------------------------------

    def compute_reliability(
        self,
        agent_id: str,
        current_list: list[str],
        prev_offers: dict,
        collective_offer_prev: list[str]
    ) -> float:
        """
        Equation (5): agent reliability d_{a,i,t}
        """
        if agent_id not in prev_offers:
            return 1.0

        prev_list = prev_offers[agent_id]
        return compute_agent_reliability(
            curr_list=current_list,
            prev_list=prev_list["valid_offers"],
            prev_coll_offer=collective_offer_prev,
            k=self.k
        )

    def compute_relevance(
        self,
        offers: list[str],
        filters: dict
    ) -> float:
        """
        Equation (4): agent relevance r_{a,i,t}
        """
        if not offers or not filters:
            return 0.0

        return compute_agent_success_scores(
            candidate_cities=offers,
            filters=filters,
            retriever=self.retriever
        )

    def compute_hallucination(
        self,
        candidates: list[str],
        rejected_globally: set[str]
    ) -> float:
        """
        Equation (6): hallucination rate h_{a,i,t}
        """
        if not candidates:
            return 0.0

        return compute_hallucination_rate(
            candidate_cities=candidates,
            catalog=self.catalog,
            rejected_cities_globally=rejected_globally
        )

    # ------------------------------------------------------------------
    # Aggregation (Equation 11)
    # ------------------------------------------------------------------

    def aggregate_scores(
        self,
        agent_outputs: dict[str, dict],
        city_scores: dict[str, float],
        prev_offers: dict[str, list[str]],
        collective_offer_prev: list[str],
        travel_filters: dict,
        rejected_cities_globally: set[str]
    ) -> tuple[dict[str, float], set[str]]:
        """
        Computes s(c,t) using Equation (11) and returns updated scores
        plus newly hallucinated cities.

        Uses self.ablated_component to determine which component to omit:
        - None: use all components (r + d - h)
        - "hallucination": omit hallucination (r + d)
        - "reliability": omit reliability (r - h)
        - "success": omit success (d - h)
        - "rank": omit rank weighting (use uniform weight)
        """
        agent_results = []
        hallucinated_cities = set()

        for agent_id, data in agent_outputs.items():
            candidates = data["candidates"]
            valid_offers = data["valid_offers"]

            # --- compute agent metrics ---
            d = self.compute_reliability(
                agent_id,
                valid_offers,
                prev_offers,
                collective_offer_prev
            )

            r = self.compute_relevance(
                valid_offers,
                travel_filters
            )

            h = self.compute_hallucination(
                candidates,
                rejected_cities_globally
            )

            # Track hallucinated items (paper definition: not in C)
            hallucinated_cities.update(
                c for c in candidates
                if c not in self.catalog
            )

            agent_results.append({
                "candidates": valid_offers,
                "reliability": d,
                "success": r,
                "hallucination": h
            })

        # --- Equation (11) update with ablation support ---
        city_scores = update_city_scores(
            cumulative_scores=city_scores,
            agent_results=agent_results,
            ablated_component=self.ablated_component
        )

        return city_scores, hallucinated_cities

