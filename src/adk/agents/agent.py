from google.genai import types
from typing import Optional, Dict, Any

from google.adk.agents.llm_agent import Agent
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv
import os, sys
from pathlib import Path
import litellm
from google.adk.models.anthropic_llm import Claude
from google.adk.models.registry import LLMRegistry

# Configure litellm globally to avoid long hangs
litellm.request_timeout = 60.0  # 60 second timeout instead of default 600s
litellm.num_retries = 5  # Retry 3 times for transient errors

# litellm._turn_on_debug()

# Add project root to sys.path to make imports work from any location
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # Navigate up to collab-rec-2026/
sys.path.insert(0, str(PROJECT_ROOT))

ENV_PATH = Path(__file__).resolve().parents[1] / ".config" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "../../../prompts/")
ENV = Environment(loader=FileSystemLoader(PROMPT_DIR))

from src.schema.agent_response import AgentResponse
from src.schema.agent_request import UserFilters, AgentRequest
from src.schema.moderator_context import ModeratorContext
from constants import CITIES
from google.adk.models.lite_llm import LiteLlm

MODEL_NAME = "gemini-2.5-flash"


async def create_specialized_agent(
    agent_name: str,
    agent_desc: str,
    model_name: str = MODEL_NAME,
    temperature: float = 0.5,
    query: Optional[str] = None,
    filters: Optional[UserFilters] = None,
    moderator_context: Optional[ModeratorContext] = None,
) -> Agent:
    template = get_prompt_template(agent_name, query, filters, moderator_context)

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=0.95,
    )

    model_name = model_name or MODEL_NAME
    model_name_lower = model_name.lower()

    def _litellm(
        model: str,
        api_base: str,
        max_tokens: int,
    ) -> LiteLlm:
        return LiteLlm(
            model=model,
            api_base=api_base,
            custom_llm_provider="openai",
            api_key=None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            extra_body={
                "guided_json": AgentResponse.model_json_schema(),
            },
        )

    if model_name_lower == "gemini":
        model = "gemini-2.5-flash"

    elif "gpt" in model_name_lower:
        model = _litellm(
            model="openai/openai/gpt-oss-20b",
            api_base="http://localhost:8000/v1",
            max_tokens=4096,
        )

    elif "gemma-12b" in model_name_lower: #running in gpu02
        model = _litellm(
            model="google/gemma-3-12b-it",
            api_base="http://localhost:8001/v1",
            max_tokens=8192,
        )

    elif "gemma-4b" in model_name_lower: #running on cuda0 in gpu01
        model = _litellm(
            model="google/gemma-3-4b-it",
            api_base="http://localhost:8000/v1",
            max_tokens=8192,
        )
    elif "olmo-7b" in model_name_lower: #running on cuda1 in gpu01
        model = _litellm(
            model="allenai/Olmo-3-7B-Think",
            api_base="http://localhost:8001/v1",
            max_tokens=8192,
        )
    elif "olmo-32b" in model_name_lower: #running on cuda0 in gpu01
        model = _litellm(
            model="allenai/Olmo-3-32B-Think",
            api_base="http://localhost:8000/v1",
            max_tokens=8192,
        )
    elif "smol-3b" in model_name_lower: #running on cuda0 in gpu01
        model = _litellm(
            model="HuggingFaceTB/SmolLM3-3B",
            api_base="http://localhost:8000/v1",
            max_tokens=8192,
        )

    elif "claude" in model_name_lower:
        os.environ["GOOGLE_CLOUD_LOCATION"] = "us-east5"
        os.environ["VERTEXAI_LOCATION"] = "us-east5"
        LLMRegistry.register(Claude)
        model = "claude-sonnet-4-5@20250929"

    else:
        model = "gemini-2.5-flash"

    return Agent(
        model=model,
        name=agent_name,
        generate_content_config=config,
        description=agent_desc,
        instruction=template,
        input_schema=AgentRequest,
        output_schema=AgentResponse,
    )

def get_prompt_template(agent_name: str, query: str,
                        filters: UserFilters | None,
                        moderator_context: Optional[ModeratorContext] = None) -> str:
    prompt_template_file = None
    match agent_name:
        case 'personalization_agent':
            filter_dict_keys = {"budget", "month", "interests"}
            prompt_template_file = 'personalization.jinja2'
        case 'sustainability_agent':
            filter_dict_keys = {"aqi", "walkability", "seasonality"}
            prompt_template_file = 'sustainability.jinja2'
        case 'sasi_agent':
            filter_dict_keys = {"aqi", "walkability", "seasonality"}
            prompt_template_file = 'sasi.jinja2'
        case 'popularity_agent':
            filter_dict_keys = {"popularity"}
            prompt_template_file = 'popularity.jinja2'
        case _:
            raise ValueError(f"Agent '{agent_name}' not recognized.")
    filters_dict = filters.model_dump(include=filter_dict_keys, exclude_none=True) if filters else {}

    # Convert moderator_context to dict if provided
    moderator_context_dict = moderator_context.model_dump() if moderator_context else None

    template = ENV.get_template(prompt_template_file).render(
        user_query=query,
        city_catalog=CITIES,
        filters=filters_dict,
        k=10,
        moderator_context=moderator_context_dict
    )
    return template
