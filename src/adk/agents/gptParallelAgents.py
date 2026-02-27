import asyncio
import copy
from google.adk.agents import BaseAgent
from google.adk.events import Event


class GPTParallelAgent(BaseAgent):
    """
    Runs sub-agents in parallel but isolates session history to prevent
    OpenAI/LiteLLM interleaving errors.
    """

    def __init__(self, name: str, sub_agents: list, description: str = ""):
        super().__init__(name=name, description=description)
        self.sub_agents = sub_agents

    async def _run_async_impl(self, ctx):
        # Helper to run an agent and collect its events independently
        async def run_isolated(agent):
            # Clone the context so each agent has its own private history 'branch'
            local_ctx = copy.copy(ctx)
            events = []
            async for event in agent.run_async(local_ctx):
                events.append(event)
            return events

        # Start all agents concurrently
        all_results = await asyncio.gather(*[run_isolated(a) for a in self.sub_agents])

        # Flatten results and yield them to the Moderator
        for agent_events in all_results:
            for event in agent_events:
                yield event
