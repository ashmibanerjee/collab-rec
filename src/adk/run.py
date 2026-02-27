from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from litellm import model_list
from typing import Optional, List
from google.genai import types
from google.adk.sessions import InMemorySessionService
import asyncio
from dotenv import load_dotenv
import json
from pathlib import Path
from src.schema.agent_response import (
   AgentResponse
)
from src.schema.agent_request import AgentRequest
from constants import CITIES
import time
import logging
from datetime import datetime

# Load .env from specific location
ENV_PATH = Path(__file__).resolve().parent / ".config" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

APP_NAME = "crs-chat_app"
USER_ID = "user_1"
SESSION_ID = "session_001"


# Session and Runner
async def _setup_session_and_runner(root_agent: Agent = None, session_id: str = SESSION_ID, request: Optional[AgentRequest] = None):
    """
    Setup ADK session and runner

    Args:
        root_agent: The agent to run
        session_id: Session identifier (defaults to SESSION_ID constant)
        request: Optional AgentRequest containing filters and other data
    """
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    # Store request data in session state if provided
    if request and request.filters:
        filters_dict = request.filters.model_dump(exclude_none=True)
        session.state["travel_filters"] = filters_dict
        print(f"Stored travel_filters in session state: {filters_dict}")
    else:
        print(f"No filters to store. request={request}, filters={request.filters if request else None}")

    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    return session, runner


# Agent Interaction
async def call_agent_async(query: str, root_agent: Agent = None, session_id: str = SESSION_ID, request: Optional[AgentRequest] = None):
    """
    Call agent asynchronously with query

    Args:
        root_agent: The agent to invoke
        session_id: Session identifier for ADK runner
        request: Optional AgentRequest containing filters and other data

    Returns:
        Tuple of (agent_name, response_text, token_count)
    """
    content = types.Content(role='user', parts=[types.Part(text=query)])
    session, runner = await _setup_session_and_runner(root_agent=root_agent, session_id=session_id, request=request)
    events = runner.run_async(user_id=USER_ID, session_id=session_id, new_message=content)

    async for event in events:
        print(f"Received event from {event.author}, is_final={event.is_final_response()}")
        if event.is_final_response():
            # Check if the response has structured output (when output_schema is used)
            # When using LiteLlm with reasoning models, structured output is in function_response
            # Try all parts to find function_response first
            print(f"Processing final response from {event.author}, number of parts: {len(event.content.parts)}")
            final_response = None

            # Debug: log all part attributes
            for idx, part in enumerate(event.content.parts):
                if hasattr(part, 'function_response'):
                    func_resp = part.function_response
                    print(f"Part {idx}: function_response type={type(func_resp)}")
                    print(f"Part {idx}: function_response dir={[a for a in dir(func_resp) if not a.startswith('_')][:20]}")

                    # Check if function_response has actual content (not None, not empty)
                    if func_resp is not None:
                        # Try to get response field from function_response
                        if hasattr(func_resp, 'response'):
                            final_response = func_resp.response
                            print(f"Using function_response.response from part {idx}, type={type(final_response)}")
                        elif hasattr(func_resp, 'name') or hasattr(func_resp, 'args'):
                            # It's a structured response object
                            final_response = func_resp
                            print(f"Using function_response object from part {idx}")
                        else:
                            final_response = func_resp
                            print(f"Using function_response directly from part {idx}")
                        break

            # If no function_response found, try text from parts
            if final_response is None:
                # Try to find JSON in any part's text (for models that don't use function_response)
                for idx, part in enumerate(event.content.parts):
                    if hasattr(part, 'text') and part.text:
                        text = part.text.strip()
                        # Try to extract JSON from text (might be wrapped in markdown or have reasoning)
                        # Look for JSON pattern: starts with { and ends with }
                        if '{' in text and '}' in text:
                            json_start = text.find('{')
                            json_end = text.rfind('}') + 1
                            if json_start >= 0 and json_end > json_start:
                                potential_json = text[json_start:json_end]
                                try:
                                    # Try parsing to validate it's JSON
                                    import json as json_lib
                                    json_lib.loads(potential_json)
                                    final_response = potential_json
                                    print(f"Found and validated JSON in part {idx} text")
                                    break
                                except:
                                    print(f"Part {idx} has braces but not valid JSON")

                # If no JSON found in any part, use first part's text
                if final_response is None:
                    part = event.content.parts[0]
                    if hasattr(part, 'text') and part.text:
                        final_response = part.text
                        print(f"Using text from first part (no JSON found), length: {len(part.text)}, preview: {part.text[:200]}")
                    else:
                        # Fallback: try to extract any available data
                        final_response = part
                        print(f"No function_response or text found, using part object: {type(part)}")

            # Extract token usage from event metadata
            token_count = None
            if hasattr(event, 'usage_metadata') and event.usage_metadata:
                usage = event.usage_metadata
                print(f"Found usage_metadata for {event.author}: {usage}")
                # Total tokens = prompt + candidates + cached
                token_count = getattr(usage, 'total_token_count', None)
                if token_count is None:
                    # Try to calculate from parts
                    prompt_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                    candidate_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                    cached_tokens = getattr(usage, 'cached_content_token_count', 0) or 0
                    token_count = prompt_tokens + candidate_tokens + cached_tokens
                    print(f"Calculated token_count: {token_count} (prompt={prompt_tokens}, candidates={candidate_tokens}, cached={cached_tokens})")
            else:
                print(f"No usage_metadata found in event for {event.author}")
                print(f"Event attributes: {dir(event)}")

            yield event.author, final_response, token_count


async def get_model_response(
        query: str,
        root_agent: Agent,
        session_id: Optional[str] = SESSION_ID,
        request: Optional[AgentRequest] = None,
        model_name: Optional[str] = None
) -> List[AgentResponse] | List[dict]:
    """
    Get model response from agent pipeline.

    Args:
        query: User query text
        root_agent: The agent to invoke
        session_id: Session identifier
        request: Optional AgentRequest containing filters and other data
    Returns:
      List of AgentResponse objects containing recommendations and explanations
    """
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())
    responses = []
    start_time = time.time()

    async for agent_name, response_text, token_count in call_agent_async(query, root_agent, session_id, request):
        end_time = time.time()
        time_taken = round(end_time - start_time, 2)
        # Parse the response_text as JSON to create AgentResponse object
        try:
            response_text = response_text.strip()
            response_text = response_text.replace("`", '')
            response_text = response_text.replace("\n", '')
            response_text = response_text.replace("json", '')
            response_data = json.loads(response_text)

            # Always set time_taken (override any existing value including None)
            response_data["time_taken"] = time_taken
            print(f"Set time_taken for {agent_name}: {time_taken}s")

            # Add token count from event metadata if not already in response
            # Priority: response_data value > event token_count
            if "total_token_count" not in response_data or response_data.get("total_token_count") is None:
                if token_count is not None:
                    response_data["total_token_count"] = token_count
                    print(f"Set total_token_count from event: {token_count}")
                else:
                    print(f"No token count available for {agent_name}")

            # Handle both "candidates" and "recommendation" keys
            # Only set item_count if not already present
            if "item_count" not in response_data:
                if "candidates" in response_data:
                    response_data["item_count"] = len(response_data["candidates"])
                elif "recommendation" in response_data:
                    response_data["item_count"] = len(response_data["recommendation"])
                    # Normalize to "candidates" key for consistency
                    response_data["candidates"] = response_data["recommendation"]
                else:
                    response_data["item_count"] = 0

            # Ensure round_number is present (should come from agent response)
            if "round_number" not in response_data:
                response_data["round_number"] = 1  # Default to 1 if not specified
            if model_name and "claude" in model_name:
                agent_response = response_data
            else:
                agent_response = AgentResponse(**response_data)
            responses.append(agent_response)
        except (json.JSONDecodeError, ValueError) as e:
            # If parsing fails, log and skip or handle as needed
            print(f"Failed to parse response from {agent_name}: {e}")
            print(f"Response type: {type(response_text)}")
            print(f"Response content: {response_text[:500]}...")
            continue

    return responses


