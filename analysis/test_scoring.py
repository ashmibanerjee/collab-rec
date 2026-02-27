from src.adk.agents.moderator_utils.compute_scores import compute_agent_success_scores, compute_agent_reliability, \
    compute_hallucination_rate, update_city_scores
from analysis.helpers import load_json, get_agent_responses
from src.k_base.context_retrieval import ContextRetrieval
from constants import CITIES as catalog


def test_city_scoring(all_agents_by_round: dict, moderator_responses: list, filters: dict):
    """
    Compute city scores across rounds for all agents.

    Args:
        all_agents_by_round: Dict mapping round_number -> list of agent responses
        moderator_responses: List of moderator responses per round
        filters: User filter constraints
    """
    retriever = ContextRetrieval()
    cumulative_scores = {}
    previous_offers = {}  # Track previous offers per agent
    collective_offer = []

    # Process each round
    for round_number in sorted(all_agents_by_round.keys()):
        agent_responses_in_round = all_agents_by_round[round_number]
        moderator_response = moderator_responses[round_number - 1] if round_number <= len(moderator_responses) else {}

        print(f"\n{'=' * 60}")
        print(f"Processing Round {round_number} - {len(agent_responses_in_round)} agents")
        print(f"{'=' * 60}")

        # Collect all agent results for this round
        agent_results_list = []

        for agent_response in agent_responses_in_round:
            agent_role = agent_response.get("agent_role")
            candidate_cities = agent_response["candidates"]

            # Compute agent success scores
            agent_success = compute_agent_success_scores(
                candidate_cities,
                filters,
                retriever
            )

            # Compute agent reliability (skip for round 1)
            if round_number != 1 and agent_role in previous_offers:
                agent_reliability = compute_agent_reliability(
                    candidate_cities,
                    previous_offers[agent_role],
                    collective_offer,
                    k=10
                )
            else:
                agent_reliability = 1.0

            # Compute hallucination rate
            agent_hallucination = compute_hallucination_rate(
                candidate_cities,
                catalog
            )

            # Prepare agent results for this agent in this round
            agent_results = {
                'candidates': candidate_cities,
                'success': agent_success,
                'reliability': agent_reliability,
                'hallucination': agent_hallucination,
            }

            agent_results_list.append(agent_results)

            # Update tracking for this agent
            previous_offers[agent_role] = candidate_cities

            print(
                f"  [{agent_role}] success={agent_success:.3f}, reliability={agent_reliability:.3f}, hallucination={agent_hallucination:.3f}")

        # Now update city scores with ALL agents' results for this round
        cumulative_scores = update_city_scores(cumulative_scores, agent_results_list)

        # Update collective offer from moderator
        rejected_cities = moderator_response.get("rejected_cities", [])
        moderator_candidates = moderator_response.get("candidates", [])
        collective_offer = moderator_candidates

        print(f"  Collective offer: {len(collective_offer)} cities")
        print(f"  Rejected cities: {len(rejected_cities)} cities")

    return cumulative_scores


def initialize_city_scores(catalog: list) -> dict:
    """Initialize city scores dictionary with default values for each city in the catalog."""
    return {city: 0 for city in catalog}


def main():
    file_name = "../data/collab-rec-2026/llm-results/gemini/mami/gemini_majority_10_rounds_old.json"
    output_data = load_json(file_name)
    for idx, output in enumerate(output_data):
        print(f"\n{'=' * 80}")
        print(f"Processing output {idx + 1} - Round 1 Analysis")
        print(f"{'=' * 80}\n")

        filters = output["query_details"]["filters"]
        agent_roles = ["personalization", "sustainability", "popularity"]

        # Organize agent responses by round
        agents_by_round = {}  # round_number -> list of agent responses
        moderator_responses = []

        # Get round 1 responses from all specialized agents
        round_number = 1
        agents_by_round[round_number] = []

        for agent_role in agent_roles:
            responses = get_agent_responses(
                agent_role=agent_role,
                data=output["response"],
                roundnr=round_number,
                all_responses=False
            )
            if responses:
                agents_by_round[round_number].append({
                    'agent_role': agent_role,
                    'candidates': responses[0].get('candidates', []),
                    'round_number': responses[0].get('round_number', round_number)
                })
                print(
                    f"[{agent_role}] Round {round_number} candidates: {len(responses[0].get('candidates', []))} cities")

        # Get moderator response for round 1
        moderator_response = get_agent_responses(
            agent_role="moderator",
            data=output["response"],
            roundnr=round_number,
            all_responses=False
        )

        if moderator_response:
            moderator_responses.append(moderator_response[0])

        # Compute city scores using all agents in round 1
        print(f"\n{'=' * 80}")
        print(f"Computing City Scores")
        print(f"{'=' * 80}")

        computed_city_scores = test_city_scoring(agents_by_round, moderator_responses, filters)

        # Compare with moderator's city_scores
        if moderator_response:
            collective_offer = moderator_response[0].get('candidates', [])
            moderator_city_scores = moderator_response[0].get('city_scores', {})
            rejected_cities = moderator_response[0].get('rejected_cities', [])

            print(f"\n{'=' * 80}")
            print(f"[Moderator] Round 1 Results:")
            print(f"{'=' * 80}")
            print(f"  Collective Offer: {len(collective_offer)} cities")
            print(f"  City Scores (from moderator): {len(moderator_city_scores)} cities")
            print(f"  Rejected Cities: {len(rejected_cities)} cities")

            # Show top 10 city scores from moderator
            if moderator_city_scores:
                sorted_scores = sorted(moderator_city_scores.items(), key=lambda x: x[1], reverse=True)
                print(f"\n  Top 10 City Scores (Moderator):")
                for i, (city, score) in enumerate(sorted_scores[:10], 1):
                    print(f"    {i}. {city}: {score:.4f}")

            # Show top 10 computed city scores
            if computed_city_scores:
                sorted_computed = sorted(computed_city_scores.items(), key=lambda x: x[1], reverse=True)
                print(f"\n  Top 10 City Scores (Computed via update_city_scores):")
                for i, (city, score) in enumerate(sorted_computed[:10], 1):
                    print(f"    {i}. {city}: {score:.4f}")

            return {
                'agents_by_round': agents_by_round,
                'moderator_response': moderator_response[0],
                'collective_offer': collective_offer,
                'moderator_city_scores': moderator_city_scores,
                'computed_city_scores': computed_city_scores,
                'rejected_cities': rejected_cities
            }
        else:
            print("ERROR: No moderator response found for round 1")
            return None


def test(output_data, source_file=""):
    # Create log file with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"../logs/convergence_analysis.log"

    # Ensure logs directory exists
    import os
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, 'w') as f:
        f.write(f"Convergence Analysis Report\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source File: {source_file}\n")
        f.write(f"{'=' * 80}\n\n")

        for idx, output in enumerate(output_data):
            query_id = output.get("query_id", f"query_{idx + 1}")
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Processing Query ID: {query_id}\n")
            f.write(f"{'=' * 80}\n\n")

            print(f"Processing Query ID: {query_id}...")  # Console feedback

            filters = output["query_details"]["filters"]
            agent_roles = ["personalization", "sustainability", "popularity"]

            # Get all responses to determine total rounds
            all_responses = output["response"]

            # Find max round number
            max_round = 0
            for response in all_responses:
                round_num = response.get("round_number", 1)
                if round_num > max_round:
                    max_round = round_num

            f.write(f"Total rounds detected: {max_round}\n\n")

            # Process each round starting from round 1
            for round_number in range(1, max_round + 1):
                f.write(f"\n{'=' * 60}\n")
                f.write(f"Round {round_number}:\n")
                f.write(f"{'=' * 60}\n")

                # Get moderator response from current round
                moderator_response = get_agent_responses(
                    agent_role="moderator",
                    data=output["response"],
                    roundnr=round_number,
                    all_responses=False
                )

                if not moderator_response:
                    f.write(f"  ⚠ No moderator response found for round {round_number}\n")
                    continue

                moderator_candidates = moderator_response[0].get("candidates", [])
                f.write(f"  Moderator's collective offer (Round {round_number}): {len(moderator_candidates)} cities\n")
                f.write(f"    Cities: {', '.join(moderator_candidates)}\n\n")

                # Get specialized agent responses from current round and compare
                f.write(f"  How many from each agent made it to moderator's offer:\n")
                for agent_role in agent_roles:
                    responses = get_agent_responses(
                        agent_role=agent_role,
                        data=output["response"],
                        roundnr=round_number,
                        all_responses=False
                    )

                    if responses:
                        agent_candidates = responses[0].get('candidates', [])
                        # Cities from this agent that made it to moderator's offer
                        contributed = list(set(agent_candidates).intersection(set(moderator_candidates)))
                        contribution_pct = (len(contributed) / len(moderator_candidates)) * 100 if moderator_candidates else 0

                        f.write(f"    [{agent_role}]: {len(contributed)}/{len(moderator_candidates)} cities "
                                f"({contribution_pct:.1f}%) from their {len(agent_candidates)} candidates made it\n")

                        # Show which cities contributed
                        if contributed:
                            f.write(f"      Contributed: {', '.join(sorted(contributed))}\n")

                        # Show which cities from agent were NOT selected
                        not_selected = list(set(agent_candidates) - set(moderator_candidates))
                        if not_selected:
                            f.write(f"      Not selected: {', '.join(sorted(not_selected))}\n")
                    else:
                        f.write(f"    [{agent_role}]: No response found\n")

                f.write("\n")  # Add spacing between rounds

            f.write(f"\n{'=' * 80}\n\n")  # Separator between queries

    print(f"\n✓ Analysis complete. Results written to: {log_file}")


MAX_ROUND = 10


def analyze_convergence_speed(output_data):
    """Check at which round agents typically converge"""
    CONFIG_ID = "c_p_27_pop_high_hard"
    output_data = [output for output in output_data if output.get("query_id") == CONFIG_ID]
    for output in output_data:
        query_id = output.get("query_id")

        # Track when moderator's offer stabilizes
        prev_offer = set()
        for round_num in range(1, MAX_ROUND + 1):
            moderator_response = get_agent_responses(
                agent_role="moderator",
                data=output["response"],
                roundnr=round_num,
                all_responses=False
            )

            if moderator_response:
                current_offer = set(moderator_response[0].get("candidates", []))
                overlap = len(prev_offer.intersection(current_offer))

                if prev_offer:
                    stability = (overlap / len(prev_offer)) * 100
                    print(f"{query_id} - Round {round_num}: {stability:.1f}% overlap with previous round")

                prev_offer = current_offer


def analyze_agent_contribution_to_moderator(output_data, log_to_file=True):
    """
    Analyze how many candidates from each specialized agent made it to the moderator's collective offer.

    Args:
        output_data: List of query outputs
        log_to_file: If True, write results to a log file
    """
    import datetime
    import os

    if log_to_file:
        log_file = f"../logs/agent_contribution_analysis.log"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        f = open(log_file, 'w')
    else:
        f = None

    def write_output(text):
        """Helper to write to both console and file"""
        print(text)
        if f:
            f.write(text + "\n")

    write_output("Agent Contribution to Moderator's Collective Offer Analysis")
    write_output(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_output("=" * 80 + "\n")

    agent_roles = ["personalization", "sustainability", "popularity"]

    for idx, output in enumerate(output_data):
        query_id = output.get("query_id", f"query_{idx + 1}")
        write_output(f"\n{'=' * 80}")
        write_output(f"Query ID: {query_id}")
        write_output(f"{'=' * 80}\n")

        # Find max round number
        all_responses = output["response"]
        max_round = max((response.get("round_number", 1) for response in all_responses), default=0)

        write_output(f"Total rounds: {max_round}\n")

        # Process each round
        for round_number in range(1, max_round + 1):
            write_output(f"\n{'-' * 60}")
            write_output(f"Round {round_number}:")
            write_output(f"{'-' * 60}")

            # Get moderator's collective offer for this round
            moderator_response = get_agent_responses(
                agent_role="moderator",
                data=output["response"],
                roundnr=round_number,
                all_responses=False
            )

            if not moderator_response:
                write_output(f"  ⚠ No moderator response found for round {round_number}")
                continue

            collective_offer = set(moderator_response[0].get("candidates", []))
            write_output(f"  Moderator's collective offer: {len(collective_offer)} cities:\n {list(collective_offer)}")

            if not collective_offer:
                write_output(f"  ⚠ Empty collective offer")
                continue

            # Check each specialized agent's contribution
            total_unique_contributed = set()
            agent_contributions = {}

            for agent_role in agent_roles:
                agent_response = get_agent_responses(
                    agent_role=agent_role,
                    data=output["response"],
                    roundnr=round_number,
                    all_responses=False
                )

                if agent_response:
                    agent_candidates = set(agent_response[0].get('candidates', []))
                    # Find which of this agent's candidates made it to collective offer
                    contributed = agent_candidates.intersection(collective_offer)
                    agent_contributions[agent_role] = contributed
                    total_unique_contributed.update(contributed)

                    contribution_pct = (len(contributed) / len(collective_offer)) * 100 if collective_offer else 0
                    write_output(f"  [{agent_role}]: {len(contributed)}/{len(collective_offer)} cities "
                               f"({contribution_pct:.1f}%) from their {len(agent_candidates)} candidates")

                    if contributed:
                        write_output(f"    Cities: {', '.join(sorted(contributed))}")
                else:
                    write_output(f"  [{agent_role}]: No response found")
                    agent_contributions[agent_role] = set()

            # Calculate overlap statistics
            write_output(f"\n  Overlap Analysis:")
            write_output(f"    Total unique cities contributed: {len(total_unique_contributed)}/{len(collective_offer)}")

            # Cities in collective offer but not from any agent (shouldn't happen ideally)
            not_from_agents = collective_offer - total_unique_contributed
            if not_from_agents:
                write_output(f"    ⚠ Cities in collective offer but not from any agent: {len(not_from_agents)}")
                write_output(f"      {', '.join(sorted(not_from_agents))}")

            # Calculate pairwise overlaps
            if len(agent_roles) >= 2:
                write_output(f"\n  Pairwise Contribution Overlap:")
                for i, role1 in enumerate(agent_roles):
                    for role2 in agent_roles[i+1:]:
                        overlap = agent_contributions[role1].intersection(agent_contributions[role2])
                        if overlap:
                            write_output(f"    [{role1}] ∩ [{role2}]: {len(overlap)} cities - {', '.join(sorted(overlap))}")

            # Cities contributed by all three agents
            if len(agent_roles) == 3:
                all_three = agent_contributions[agent_roles[0]].intersection(
                    agent_contributions[agent_roles[1]],
                    agent_contributions[agent_roles[2]]
                )
                if all_three:
                    write_output(f"\n  Cities contributed by ALL agents: {len(all_three)}")
                    write_output(f"    {', '.join(sorted(all_three))}")

        write_output(f"\n{'=' * 80}\n")

    if f:
        f.close()
        print(f"\n✓ Analysis complete. Results written to: {log_file}")


if __name__ == "__main__":

    file_name = "../data/collab-rec-2026/llm-results/gemini/mami/gemini_aggressive_10_rounds_fewshot_v1.json"
    output_data = load_json(file_name)
    CONFIG_ID = "c_p_27_pop_high_hard"
    output_data = [output for output in output_data if output.get("query_id") == CONFIG_ID]
    test(output_data, source_file=file_name)
    # analyze_convergence_speed(output_data)
    analyze_agent_contribution_to_moderator(output_data, log_to_file=True)
