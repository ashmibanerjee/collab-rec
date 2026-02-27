import sys
from pathlib import Path

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm  # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from pydantic import BaseModel

from adk.agents.agent import create_specialized_agent
from schema.agent_request import UserFilters

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # Navigate up to collab-rec-2026/
sys.path.insert(0, str(PROJECT_ROOT))

ENV_PATH = Path(__file__).resolve().parents[1] / ".config" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

MODEL_GPT_4O = "openai/gpt-4o"

from pydantic import BaseModel, Field
from typing import Optional
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from pydantic import BaseModel, Field
from google.genai import types
from google.adk.models.llm_request import LlmRequest


class WeatherResponse(BaseModel):
    temperature: int = Field(description="Temperature in Fahrenheit")
    conditions: str = Field(description="Weather conditions")




async def create_session_gpt():
    print("[DEBUG] Creating agent...")
    # Create an agent powered by OpenAI's GPT model
#     agent = Agent(
#     name="test",
#     instruction="Output a name as 'John' and age as 30",
#     model=LiteLlm(model="gpt-4o-mini"),
#     output_schema=User,
#     output_key="user",
# )
    request = {
  "config_id": "c_p_1_pop_medium_sustainable",
  "filters": {
    "popularity": "medium",
    "month": "March",
    "budget": "medium",
    "walkability": "great"
  },
  "query": "Great walkable European city with medium budget for March."
}
    filters = request["filters"]
    llm_request = LlmRequest(
        model="gemini-2.0-flash",
        contents=[...],
        config=types.GenerateContentConfig(
            response_schema=WeatherResponse
        )
    )
    agent = LlmAgent(
        model=LiteLlm(model="openai/gpt-4o"),
        name="weather_agent",
        instruction="Extract weather information and respond in the specified JSON format",
        output_schema=WeatherResponse,
    )

    print("[DEBUG] Agent created")

    # Set up session and runner
    print("[DEBUG] Creating session service...")
    session_service_gpt = InMemorySessionService()
    print("[DEBUG] Creating session...")
    session_gpt = await session_service_gpt.create_session(
        app_name="gpt_app",
        user_id="user_1",
        session_id="session_gpt"
    )
    print("[DEBUG] Session created")

    print("[DEBUG] Creating runner...")
    runner_gpt = Runner(
        agent=agent,
        app_name="gpt_app",
        session_service=session_service_gpt
    )
    print("[DEBUG] Runner created")
    return session_gpt, runner_gpt


async def call_agent_async(query: str, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    print("[DEBUG] Preparing content...")
    content = types.Content(role='user', parts=[types.Part(text=query)])
    print("[DEBUG] Creating session...")
    session, runner = await create_session_gpt()
    print("[DEBUG] Running agent...")
    final_response_text = "Agent did not produce a final response."
    events = runner.run_async(user_id=user_id, session_id=session_id, new_message=content)

    print("[DEBUG] Iterating events...")
    # Execute the agent and find the final response
    async for event in events:
        print(f"[DEBUG] Got event: {type(event)}")
        if event.is_final_response():
            print("[DEBUG] Found final response")
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
                print(f"<<< Agent Response: {final_response_text}")
                return event.author, final_response_text

    print(f"<<< Agent Response: {final_response_text}")
    return None, final_response_text


# Test the GPT agent
async def test_gpt_agent():
    print("\n--- Testing GPT Agent ---")
    author, response = await call_agent_async(
        "Suggest places to visit in Paris.",
        user_id="user_1",
        session_id="session_gpt"
    )
    print(f"\nFinal result from {author}: {response}")


# Or if running as a standard Python script:
if __name__ == "__main__":
    async def main():
        await test_gpt_agent()


    asyncio.run(main())
