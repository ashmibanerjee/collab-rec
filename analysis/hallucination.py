from analysis.helpers import load_json, get_agent_responses
from constants import CITIES


def compute_hallucination_rate(agent_data, moderator_data):
    """
    Computes hallucination rate for each round.
    Returns a dict mapping round_nr -> hallucination_rate
    """
    hallucination_rate = {}

    for round_data in agent_data:
        round_nr = round_data.get("round_number", None)
        if round_nr is None:
            print(f"Warning: Round missing round_number, skipping")
            continue

        candidates = round_data.get("candidates", [])
        total_candidates = len(candidates)
        hallucination_count = 0
        if round_nr == 1: continue
        rejected_cities, collective_offer = get_rejected_cities_for_round(moderator_data, round_nr)
        print(
            f"\nRound {round_nr}:, rejected cities: {rejected_cities}; candidates: {candidates}\n collective offer: {collective_offer}")
        for candidate in candidates:
            if candidate not in CITIES:
                if candidate in rejected_cities:
                    print(f"  Hallucinated city: {candidate}")
                hallucination_count += 1

        # Calculate rate (0.0 to 1.0)
        rate = hallucination_count / total_candidates if total_candidates > 0 else 0.0
        hallucination_rate[round_nr] = rate

        print(f"  Total hallucinations: {hallucination_count}/{total_candidates} (rate: {rate:.3f})")

    return hallucination_rate


def get_rejected_cities_for_round(all_rounds_data, round_nr: int):
    """
    Extracts rejected cities from the moderator response for a specific round.

    Args:
        all_rounds_data: List of all round data dictionaries
        round_nr: The round number to get rejected cities for
    """
    moderator_response = get_agent_responses(
        agent_role="moderator",
        data=all_rounds_data,
        roundnr=round_nr,
        all_responses=False
    )
    rejected_cities = moderator_response[0].get("rejected_cities", []) if moderator_response else []
    collective_offer = moderator_response[0].get("candidates", []) if moderator_response else []
    return rejected_cities, collective_offer


def hallucination_analysis(agent_role: str = "personalization"):
    file_name = ("../data"
                 "/collab-rec-2026/llm-results/gemini/mami/gemini_majority_10_rounds.json")
    output_data = load_json(file_name)

    for idx, output in enumerate(output_data):
        print(f"\n{'=' * 60}")
        print(f"Processing output {idx + 1}")
        print(f"{'=' * 60}")

        agent_responses = get_agent_responses(agent_role=agent_role,
                                              data=output["response"],
                                              roundnr=10, all_responses=True)
        moderator_responses = get_agent_responses(agent_role="moderator", data=output["response"], roundnr=10,
                                                  all_responses=True)

        hallucination_rates = compute_hallucination_rate(agent_responses, moderator_responses)

        print(f"\n{'=' * 60}")
        print(f"Summary - Hallucination Rates by Round:")
        print(f"{'=' * 60}")
        for round_nr in sorted(hallucination_rates.keys()):
            rate = hallucination_rates[round_nr]
            # print(f"  Round {round_nr}: {rate:.3f} ({rate * 100:.1f}%)")

        avg_rate = sum(hallucination_rates.values()) / len(hallucination_rates) if hallucination_rates else 0
        print(f"\n  Average hallucination rate: {avg_rate:.3f} ({avg_rate * 100:.1f}%)")
        print(f"{'=' * 60}\n")
        break  # TODO remove


if __name__ == "__main__":
    hallucination_analysis("personalization")
    # hallucination_analysis("sustainability")
    # hallucination_analysis("popularity")
