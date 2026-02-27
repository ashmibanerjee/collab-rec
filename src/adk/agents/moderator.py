from typing import List, Dict, Any, AsyncGenerator, Optional
import logging

from google.adk.agents import BaseAgent
from google.adk.events import Event
from typing_extensions import override

from src.adk.agents.moderator_utils.response_parser import ResponseParser
from src.adk.agents.moderator_utils.scoring_manager import ScoringManager
from src.adk.agents.moderator_utils.feedback_manager import FeedbackManager
from src.adk.agents.moderator_utils.rejection_manager import RejectionManager, remove_rejected_from_scores
from src.adk.agents.moderator_utils.aggregation_manager import AggregationManager
from src.adk.agents.moderator_utils.early_stopping_manager import EarlyStoppingManager
from src.adk.agents.moderator_utils.agent_recreation_manager import AgentRecreationManager
from src.adk.agents.moderator_utils.session_state_manager import SessionStateManager
from src.schema.moderator_context import ModeratorContext
from src.schema.agent_response import AgentResponse
from constants import CITIES

logger = logging.getLogger(__name__)


class ModeratorAgent(BaseAgent):
    """
    Fully self-contained ADK-native Collab-REC Moderator
    """

    def __init__(
            self,
            parallel_agent,
            all_cities: List[str] = CITIES,
            k: int = 10,
            rejection_method: str = "majority",
            min_rounds: int = 5,
            early_stop_thresholds=(0.2, 0.4, 0.6, 0.8),
            early_stopping_threshold: float = None,  # Actual threshold to stop at (None = MN, 0.2 = M20, 0.6 = M60)
            smoothing_weight: float | None = None,
            standardize: bool = False,
            travel_filters: Dict[str, Any] = None,
            request: Any = None,  # Store original request for recreating agents
            retriever=None,  # Knowledge base retriever for scoring
            ablated_component: str | None = None
    ):
        super().__init__(
            name="ModeratorAgent",
            description="ADK-native Collab-REC moderator with feedback and early stopping"
        )

        self._parallel_agent = parallel_agent
        self._request = request  # Store original request
        self._k = k
        self._travel_filters = travel_filters or {}
        self._all_cities = all_cities
        self._retriever = retriever

        # Initialize managers (private attributes to avoid ADK serialization)
        self._response_parser = ResponseParser()
        self._scoring_manager = ScoringManager(k=k, retriever=retriever, ablated_component=ablated_component)
        self._feedback_manager = FeedbackManager(k=k)
        self._rejection_manager = RejectionManager(rejection_method=rejection_method)
        self._aggregation_manager = AggregationManager(k=k, standardize=standardize)
        self._early_stopping_manager = EarlyStoppingManager(
            min_rounds=min_rounds,
            early_stop_thresholds=early_stop_thresholds,
            early_stopping_threshold=early_stopping_threshold,  # Pass the actual stopping threshold
            retriever=retriever
        )
        self._agent_recreation_manager = AgentRecreationManager()
        self._session_state_manager = SessionStateManager()

        # State variables
        self._round = 0
        self._city_scores = {c: 0.0 for c in all_cities}
        self._rejected_cities_globally = set()  # Cities that should not be recommended (hallucinated + rejected) - cumulative across all rounds
        self._collective_offer: List[str] = []
        self._round_1_collective_offer: List[
            str] = []  # Store Round 1 (first round) collective offer for improvement calculation
        self._prev_offers: Dict[str, Dict[str, Any]] = {}

    async def _recreate_parallel_agents_with_feedback(self, agent_contexts: Dict[str, ModeratorContext]):
        """
        Recreate parallel agents with updated moderator context/feedback.
        This allows agents to receive feedback in their prompts for subsequent rounds.
        """
        return await self._agent_recreation_manager.recreate_parallel_agents_with_feedback(
            agent_contexts, self._request
        )

    def _enhance_agent_event_with_computed_metrics(
            self,
            event: Event,
            total_token_count: Optional[int],
            agent_metrics: Dict[str, Any]
    ) -> Event:
        """
        Enhance individual agent event with computed metrics from scoring manager.

        Args:
            event: Original event from agent
            total_token_count: Token count extracted from usage_metadata
            agent_metrics: Computed metrics from parsed dictionary (includes reliability_score,
                          relevance_score, hallucination_rate)

        Returns:
            Enhanced event with actual computed metrics added to response
        """
        from google.genai import types as genai_types
        import json

        # Extract response text from event
        if not hasattr(event, 'content') or not event.content:
            return event

        for part in event.content.parts:
            if hasattr(part, 'text') and part.text:
                try:
                    # Parse the response JSON
                    response_text = part.text.strip()
                    response_text = response_text.replace("`", '')
                    response_text = response_text.replace("\n", '')
                    response_text = response_text.replace("json", '')

                    response_data = json.loads(response_text)

                    # Add total_token_count from usage_metadata
                    if total_token_count is not None:
                        response_data["total_token_count"] = total_token_count

                    # Add rejections from parsed data (cities this agent rejected from collective offer)
                    response_data["rejections"] = agent_metrics.get("rejections", None)

                    # Create enhanced response text
                    enhanced_text = json.dumps(response_data, indent=2)

                    # Create new event with enhanced content
                    enhanced_event = Event(
                        author=event.author,
                        content=genai_types.Content(
                            role="model",
                            parts=[genai_types.Part(text=enhanced_text)]
                        )
                    )

                    return enhanced_event

                except (json.JSONDecodeError, Exception) as e:
                    print(f"[{self.name}] Failed to enhance event from {event.author}: {e}")
                    return event

        return event

    @override
    async def _run_async_impl(
            self,
            ctx: Any
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic for the Collab-REC workflow.
        Coordinates parallel agents, aggregates recommendations, and provides feedback.
        """
        self._round += 1

        print(f"\n{'=' * 80}")
        print(f"[{self.name}] 🔄 STARTING ROUND {self._round}")
        print(f"{'=' * 80}")

        # Get travel filters from session state first, fallback to instance variable
        travel_filters = ctx.session.state.get("travel_filters", self._travel_filters)
        print(f"[{self.name}] Retrieved travel_filters: {travel_filters}")

        # Store in session state if not already there
        if "travel_filters" not in ctx.session.state and travel_filters:
            ctx.session.state["travel_filters"] = travel_filters

        # ----------------------------------------------------
        # 1. Recreate agents with feedback for rounds > 1
        # ----------------------------------------------------
        if self._round > 1 and self._request:
            agent_contexts_dict = ctx.session.state.get("agent_contexts", {})

            if agent_contexts_dict:
                print(f"[{self.name}] Found agent contexts from previous round, recreating agents with feedback...")
                print(f"[{self.name}] DEBUG: agent_contexts_dict type: {type(agent_contexts_dict)}")

                # Convert dict to ModeratorContext objects
                agent_contexts = {}
                for agent_name, context_data in agent_contexts_dict.items():
                    print(f"[{self.name}] DEBUG: Processing {agent_name}, context_data type: {type(context_data)}")

                    # Session state should always store dicts, not ModeratorContext objects
                    if isinstance(context_data, dict):
                        try:
                            agent_contexts[agent_name] = ModeratorContext(**context_data)
                            print(f"[{self.name}] ✓ Created ModeratorContext for {agent_name}")
                        except Exception as e:
                            print(f"[{self.name}] ERROR: Failed to create ModeratorContext for {agent_name}: {e}")
                            print(
                                f"[{self.name}] DEBUG: context_data keys: {context_data.keys() if isinstance(context_data, dict) else 'N/A'}")
                            raise
                    elif isinstance(context_data, ModeratorContext):
                        print(
                            f"[{self.name}] WARNING: {agent_name} context is already a ModeratorContext object, using directly")
                        agent_contexts[agent_name] = context_data
                    else:
                        print(f"[{self.name}] ERROR: Unexpected context type for {agent_name}: {type(context_data)}")
                        raise TypeError(f"Expected dict or ModeratorContext, got {type(context_data)}")

                # Recreate parallel agents with feedback
                self._parallel_agent = await self._recreate_parallel_agents_with_feedback(agent_contexts)
            else:
                print(f"[{self.name}] WARNING: No agent contexts found in session state for round {self._round}")

        # ----------------------------------------------------
        # 2. Inject moderator context for this round
        # ----------------------------------------------------
        self._session_state_manager.inject_moderator_context(
            ctx, self._round, self._collective_offer,
            self._rejected_cities_globally, self._prev_offers
        )

        # ----------------------------------------------------
        # 3. Run specialized agents in parallel
        # ----------------------------------------------------
        print(f"[{self.name}] Running parallel specialized agents (round {self._round})...")

        raw_responses = {}
        agent_events = {}  # Store events to yield after computing metrics

        async for event in self._parallel_agent.run_async(ctx):
            # Collect responses from parallel agents
            if event.is_final_response():
                # Extract total_token_count from usage_metadata
                total_token_count = None
                if hasattr(event, 'usage_metadata') and event.usage_metadata:
                    total_token_count = getattr(event.usage_metadata, 'total_token_count', None)
                    print(f"[{self.name}] Extracted token count from {event.author}: {total_token_count}")

                # Store event for later enhancement (after metrics are computed)
                agent_events[event.author] = (event, total_token_count)

                # Extract response data from the event
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            # print(f"[{self.name}] Parsing response from {event.author}: {part.text[:200]}...")
                            # Parse agent response
                            parsed_response = self._response_parser.parse_agent_response(part.text)
                            candidates = parsed_response.get("candidates", [])
                            feedback_acknowledged = parsed_response.get("feedback_acknowledged", None)

                            if candidates:
                                raw_responses[event.author] = {
                                    "candidates": candidates,
                                    "feedback_acknowledged": feedback_acknowledged,
                                    "total_token_count": total_token_count
                                }
                                print(
                                    f"[{self.name}] Successfully parsed {len(candidates)} candidates from {event.author}")
                                print(f"[{self.name}] Feedback acknowledged by {event.author}: {feedback_acknowledged}")
                                if total_token_count:
                                    print(f"[{self.name}] Total tokens used by {event.author}: {total_token_count}")
                            else:
                                print(f"[{self.name}] WARNING: No candidates parsed from {event.author}")
                                raw_responses[event.author] = {
                                    "candidates": [],
                                    "feedback_acknowledged": feedback_acknowledged,
                                    "total_token_count": total_token_count
                                }

        print(f"[{self.name}] Collected {len(raw_responses)} agent responses")

        # ----------------------------------------------------
        # 3a. EXPLICIT CHECK: Ensure all 3 specialized agents have responded
        # ----------------------------------------------------
        required_agents = {'personalization_agent', 'sustainability_agent', 'popularity_agent'}
        present_agents = set(raw_responses.keys())

        if not required_agents.issubset(present_agents):
            missing_agents = required_agents - present_agents
            error_msg = f"Not all specialized agents have completed Round {self._round}. Missing: {sorted(missing_agents)}"
            print(f"[{self.name}] ❌ ERROR: {error_msg}")
            print(f"[{self.name}] Expected agents: {sorted(required_agents)}")
            print(f"[{self.name}] Received agents: {sorted(present_agents)}")
            raise ValueError(error_msg)

        print(f"[{self.name}] ✓ All {len(required_agents)} specialized agents have provided their recommendations")
        print(f"[{self.name}] Agents: {sorted(present_agents)}")

        # ----------------------------------------------------
        # 4. Parse all responses
        # ----------------------------------------------------
        parsed = self._response_parser.parse_all_responses(
            raw_responses, self._city_scores, self._rejected_cities_globally
        )

        # Log feedback acknowledgement summary
        self._log_feedback_acknowledgement(parsed)

        # ----------------------------------------------------
        # 5. Compute rejections
        # ----------------------------------------------------
        print(f"[{self.name}] Computing rejections...")
        self._rejected_cities_globally = self._rejection_manager.compute_rejections(
            parsed, self._collective_offer, self._rejected_cities_globally, self._round
        )

        # Remove rejected cities from scoring
        self._city_scores = remove_rejected_from_scores(
            self._city_scores, self._rejected_cities_globally
        )

        # ----------------------------------------------------
        # 6. Scoring and aggregation
        # ----------------------------------------------------
        print(f"[{self.name}] Computing scores for recommendations...")
        self._city_scores, hallucinated_cities = self._scoring_manager.aggregate_scores(
            parsed, self._city_scores, self._prev_offers,
            self._collective_offer, travel_filters, self._rejected_cities_globally
        )

        # Add hallucinated cities to rejected_cities_globally (cumulative blacklist)
        if hallucinated_cities:
            print(
                f"[{self.name}] Detected {len(hallucinated_cities)} hallucinated cities: {sorted(hallucinated_cities)}")
            self._rejected_cities_globally.update(hallucinated_cities)
            # Remove hallucinated cities from city scores
            for city in hallucinated_cities:
                if city in self._city_scores:
                    del self._city_scores[city]

        # ----------------------------------------------------
        # 6a. Now yield enhanced agent events with computed metrics
        # ----------------------------------------------------
        print(f"[{self.name}] Yielding enhanced agent responses with computed metrics...")
        print(f"[{self.name}] DEBUG: agent_events keys: {list(agent_events.keys())}")
        print(f"[{self.name}] DEBUG: parsed keys: {list(parsed.keys())}")

        for agent_name, (event, total_token_count) in agent_events.items():
            agent_metrics = parsed.get(agent_name, {})

            # Enhance event with the NOW-COMPUTED metrics from parsed dict
            enhanced_event = self._enhance_agent_event_with_computed_metrics(
                event, total_token_count, agent_metrics
            )
            yield enhanced_event

        # ----------------------------------------------------
        # 7. Normalize and create collective offer
        # ----------------------------------------------------
        print(f"[{self.name}] Normalizing scores and creating collective offer...")
        self._city_scores = self._aggregation_manager.normalize_scores(self._city_scores)
        self._collective_offer = self._aggregation_manager.create_collective_offer(self._city_scores)

        # Store Round 1 collective offer for improvement calculation baseline
        if self._round == 1:
            self._round_1_collective_offer = self._collective_offer.copy()
            print(
                f"[{self.name}] Stored Round 1 collective offer (baseline): {self._round_1_collective_offer[:3]}... (len={len(self._round_1_collective_offer)})")

        # ----------------------------------------------------
        # 8. Generate feedback
        # ----------------------------------------------------
        print(f"[{self.name}] Generating feedback for agents...")
        agent_feedback = self._feedback_manager.generate_all_feedback(
            parsed, self._collective_offer,
            lambda candidates: self._scoring_manager.compute_hallucination(
                candidates, self._city_scores
            )
        )

        # ----------------------------------------------------
        # 10. Update session state and prepare context for next round
        # ----------------------------------------------------
        self._prev_offers = parsed

        # Create agent-specific moderator contexts for the next round
        agent_contexts = self._session_state_manager.prepare_agent_contexts_for_next_round(
            self._round, self._collective_offer, self._rejected_cities_globally,
            parsed, agent_feedback
        )

        # Store results in session state
        self._session_state_manager.update_session_state(
            ctx, self._round, self._collective_offer, self._rejected_cities_globally,
            parsed, agent_feedback, agent_contexts, self._city_scores,
            self._early_stopping_manager.get_threshold_status()
        )

        print(f"[{self.name}] Workflow finished. Collective offer: {self._collective_offer}")

        if ctx.session.state.get('agent_contexts'):
            print(f"[{self.name}] Agent context keys: {list(ctx.session.state['agent_contexts'].keys())}")

        print(f"{'=' * 80}\n")

        yield self._create_moderator_response(parsed)

        print(f"[{self.name}] → Continuing to next round (round {self._round + 1})...")

    def _log_feedback_acknowledgement(self, parsed: Dict[str, Dict[str, Any]]) -> None:
        """Log feedback acknowledgement summary."""
        feedback_summary = {agent: data.get("feedback_acknowledged") for agent, data in parsed.items()}
        print(f"[{self.name}] Feedback acknowledgement summary: {feedback_summary}")

        if self._round > 1:
            agents_that_acknowledged = [agent for agent, ack in feedback_summary.items() if ack is True]
            agents_that_didnt = [agent for agent, ack in feedback_summary.items() if ack is not True]

            if agents_that_acknowledged:
                print(f"[{self.name}] ✓ Agents that acknowledged feedback: {agents_that_acknowledged}")
            if agents_that_didnt:
                print(f"[{self.name}] ✗ Agents that didn't acknowledge feedback: {agents_that_didnt}")

    def _create_moderator_response(self, parsed: Dict[str, Dict[str, Any]]) -> Event:
        """
        Create moderator response event with aggregated metrics.

        Args:
            parsed: Parsed responses from all agents

        Returns:
            Event containing moderator's response
        """
        from google.genai import types as genai_types

        print(f"[{self.name}] Presenting moderator's fused recommendations for round {self._round}...")

        # Calculate total tokens used by all agents
        total_tokens = sum(
            data.get("total_token_count", 0) or 0
            for data in parsed.values()
        )
        # Extract non-zero city scores
        non_zero_city_scores = self._aggregation_manager.get_non_zero_scores(self._city_scores)
        print(f"[{self.name}] Non-zero city scores: {len(non_zero_city_scores)} cities")
        print(
            f"[{self.name}] Rejected cities globally (hallucinated + rejected by agents): {len(self._rejected_cities_globally)} - {sorted(self._rejected_cities_globally)}")

        # Build AgentResponse - ensure values are properly converted
        agent_response = AgentResponse(
            agent_role="moderator",
            candidates=self._collective_offer,
            explanation=f"Top {self._k} cities aggregated from all agents in round {self._round}.",
            trade_off=f"Rejected/filtered {len(self._rejected_cities_globally)} cities total. Round {self._round} complete.",
            feedback_acknowledged=True,
            round_number=self._round,
            item_count=len(self._collective_offer),
            total_token_count=int(total_tokens) if total_tokens > 0 else 0,
            rejections=None,  # Moderator doesn't reject from collective offer
            city_scores=non_zero_city_scores,
            rejected_cities=list(self._rejected_cities_globally),  # Hallucinated + rejected cities (cumulative)
            time_taken=None,
        )

        print(f"[{self.name}] DEBUG: After creating AgentResponse:")
        print(f"  agent_response.total_token_count={agent_response.total_token_count}")

        # Convert to JSON
        response_text = agent_response.model_dump_json(indent=2, exclude_none=False)

        print(f"[{self.name}] ====== MODERATOR'S FINAL RESPONSE (with metrics) ======")
        print(f"[{self.name}] This is the aggregated response from the moderator")
        print(f"[{self.name}] Total tokens used by all agents this round: {total_tokens}")

        # Create response event
        moderator_event = Event(
            author=self.name,
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text=response_text)]
            )
        )

        print(f"[{self.name}] ====== END MODERATOR'S FINAL RESPONSE ======")

        return moderator_event
