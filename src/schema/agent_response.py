from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from constants import CITIES
import logging

logger = logging.getLogger(__name__)


class AgentResponse(BaseModel):
    # 1. CORE FIELDS (Always required)
    agent_role: Literal["personalization", "sustainability", "popularity", "moderator", "sasi"] = Field(
        ...,
        description="Role of the agent performing the task"
    )

    candidates: List[str] = Field(
        ...,
        description="List of recommended European city names from the allowed catalog."
    )

    explanation: Optional[str] = Field(
        ...,
        description="Overall justification (max 200 chars). Provide null if not applicable."
    )

    # 2. FEEDBACK & ROUND FIELDS
    trade_off: Optional[str] = Field(
        ...,
        description="Trade-offs made (max 200 chars). Provide null if not applicable."
    )

    feedback_acknowledged: Optional[bool] = Field(
        ...,
        description="True if feedback was incorporated. Provide null if first round."
    )

    round_number: Optional[int] = Field(
        ...,
        description="The current negotiation round number."
    )

    # 3. METRICS & REJECTIONS
    time_taken: Optional[float] = Field(
        ...,
        description="Time in seconds. Usually handled by system, provide null."
    )

    item_count: Optional[int] = Field(
        ...,
        description="Number of cities in the candidates list."
    )

    total_token_count: Optional[int] = Field(
        ...,
        description="Total tokens used. Provide null if unknown."
    )

    rejections: Optional[List[str]] = Field(
        ...,
        description="Cities rejected from the collective offer. Provide empty list or null if none."
    )

    # 4. MODERATOR SPECIFIC FIELDS
    city_scores: Optional[Dict[str, float]] = Field(
        ...,
        description="FOR MODERATOR ONLY: City score mapping. Others provide null."
    )

    rejected_cities: Optional[List[str]] = Field(
        ...,
        description="FOR MODERATOR ONLY: Cumulative blacklist. Others provide null."
    )

    @field_validator("candidates")
    @classmethod
    def check_cities(cls, v):
        """Log invalid cities but DO NOT raise error or filter."""
        for city in v:
            if city not in CITIES:
                # We log it as a warning but return it anyway for the Moderator to handle
                logger.warning(f"Hallucination Detected: City '{city}' is not in catalog.")
        return v
    # #TODO: maybe bring it back for gpt
    # model_config = {
    #     "json_schema_extra": {
    #         "additionalProperties": False
    #     }
    # }

