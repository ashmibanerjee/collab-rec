"""
Early stopping logic for collaborative recommendation.
"""
from typing import Dict, Any, List, Optional
import logging

from adk.agents.moderator_utils.compute_scores import compute_agent_success_scores

logger = logging.getLogger(__name__)


class EarlyStoppingManager:
    """Manages early stopping logic based on improvement metrics."""

    def __init__(
        self,
        min_rounds: int = 5,
        early_stop_thresholds: tuple = (0.2, 0.4, 0.6, 0.8),
        early_stopping_threshold: float = None,
        retriever = None
    ):
        """
        Initialize early stopping manager.

        Args:
            min_rounds: Minimum number of rounds before early stopping
            early_stop_thresholds: Thresholds to track (for reporting)
            early_stopping_threshold: Actual threshold to stop at (e.g., 0.2 for M20, 0.6 for M60, None for MN)
            retriever: Knowledge base retriever for success scoring
        """
        self.min_rounds = min_rounds
        self.early_stop_thresholds: Dict[float, int | None] = {
            t: None for t in early_stop_thresholds
        }
        self.early_stopping_threshold = early_stopping_threshold  # Actual stopping threshold (None = no early stopping)
        self.baseline_success = None
        self.retriever = retriever

    def compute_improvement(
        self,
        collective_offer: List[str],
        travel_filters: Dict[str, Any],
        round_number: int,
        round_1_offer: List[str] = None
    ) -> float:
        """
        Compute improvement score compared to Round 1 (initial collective offer).

        Args:
            collective_offer: Current collective offer
            travel_filters: User travel filters
            round_number: Current round number
            round_1_offer: Round 1 collective offer (for baseline comparison)

        Returns:
            Improvement score (0.0 to 1.0+)
        """
        print(f"\n[EarlyStoppingManager] compute_improvement called:")
        print(f"  round_number: {round_number}")
        print(f"  min_rounds: {self.min_rounds}")
        print(f"  collective_offer: {collective_offer[:3]}... (len={len(collective_offer)})")
        print(f"  travel_filters: {travel_filters}")
        print(f"  retriever: {self.retriever is not None}")

        # Don't compute improvement before minimum rounds
        if round_number < self.min_rounds:
            print(f"  → Skipping: round {round_number} < min_rounds {self.min_rounds}")
            return 0.0

        # Establish Round 1 baseline on first eligible round
        if self.baseline_success is None:
            print(f"  → Establishing Round 1 baseline")
            # Use round_1_offer if provided, otherwise use current offer as baseline
            baseline_offer = round_1_offer if round_1_offer else collective_offer

            if self.retriever is not None and travel_filters:
                self.baseline_success = compute_agent_success_scores(
                    baseline_offer, travel_filters, self.retriever
                )
                print(f"  → Round 1 baseline computed: {self.baseline_success:.3f}")
            else:
                # Fallback: assume baseline if no retriever
                self.baseline_success = 1.0
                print(f"  → Round 1 baseline fallback (no retriever/filters): {self.baseline_success}")
            logger.info(f"Round 1 baseline established: {self.baseline_success:.3f}")

        # Compute current round success
        print(f"  → Computing current round success...")
        if self.retriever is not None and travel_filters:
            round_score = compute_agent_success_scores(
                collective_offer, travel_filters, self.retriever
            )
            print(f"  → Current round score: {round_score:.3f}")
        else:
            # Fallback: assume no improvement if no retriever
            round_score = self.baseline_success
            print(f"  → Current round score fallback (no retriever/filters): {round_score}")

        # Perfect score = maximum improvement
        if round_score == 1:
            logger.info("Perfect success score achieved (1.0)")
            print(f"  → Perfect score achieved!")
            return 1.0

        # Calculate improvement relative to Round 1: (round_score - round_1) / round_1
        try:
            improvement = max(0.0, (round_score - self.baseline_success) / self.baseline_success)
        except ZeroDivisionError:
            improvement = 0.0
            print(f"  → ZeroDivisionError: Round 1 baseline is 0, setting improvement to 0")

        print(f"  → Improvement calculation:")
        print(f"     round_score (current): {round_score:.3f}")
        print(f"     round_1 (baseline): {self.baseline_success:.3f}")
        print(f"     improvement: {improvement:.3f} = ({round_score:.3f} - {self.baseline_success:.3f}) / {self.baseline_success:.3f}")

        logger.info(
            f"Improvement: {improvement:.3f} "
            f"(round_score={round_score:.3f}, round_1={self.baseline_success:.3f})"
        )

        return improvement

    def update_thresholds(
        self,
        improvement: float,
        round_number: int
    ) -> None:
        """
        Update early stopping thresholds based on improvement.

        Args:
            improvement: Current improvement score
            round_number: Current round number
        """
        print(f"\n[EarlyStoppingManager] update_thresholds called:")
        print(f"  round_number: {round_number}")
        print(f"  improvement: {improvement:.3f}")
        print(f"  min_rounds: {self.min_rounds}")
        print(f"  Current threshold status: {self.early_stop_thresholds}")

        for threshold in self.early_stop_thresholds:
            print(f"  Checking threshold {threshold}:")
            print(f"    round >= min_rounds: {round_number >= self.min_rounds}")
            print(f"    improvement > threshold: {improvement} > {threshold} = {improvement > threshold}")
            print(f"    threshold not yet reached: {self.early_stop_thresholds[threshold] is None}")

            if (round_number >= self.min_rounds and
                improvement > threshold and
                self.early_stop_thresholds[threshold] is None):

                self.early_stop_thresholds[threshold] = round_number
                print(f"    ✓ Threshold {threshold} REACHED at round {round_number}")
                logger.info(
                    f"Early stopping threshold {threshold} reached at round {round_number}"
                )
            else:
                print(f"    ✗ Threshold {threshold} not reached")

        print(f"  Updated threshold status: {self.early_stop_thresholds}")

    def should_stop(
        self,
        improvement: float,
        round_number: int
    ) -> bool:
        """
        Determine if early stopping condition is met.

        Args:
            improvement: Current improvement score
            round_number: Current round number

        Returns:
            True if should stop, False otherwise
        """
        # Must meet minimum rounds requirement
        if round_number < self.min_rounds:
            return False

        # If no early stopping threshold is set (MN configuration), never stop early
        if self.early_stopping_threshold is None:
            return False

        # Check if improvement exceeds the configured early stopping threshold
        should_stop = improvement >= self.early_stopping_threshold

        if should_stop:
            logger.info(
                f"Early stopping condition met: "
                f"improvement ({improvement:.3f}) >= threshold ({self.early_stopping_threshold})"
            )

        return should_stop

    def get_threshold_status(self) -> Dict[float, int | None]:
        """
        Get status of all early stopping thresholds.

        Returns:
            Dictionary mapping thresholds to rounds when they were met (or None)
        """
        return self.early_stop_thresholds.copy()

