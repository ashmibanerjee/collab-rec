from typing import Optional

from pydantic import BaseModel, Field, field_validator

from src.schema.moderator_context import ModeratorContext


class UserFilters(BaseModel):
    popularity: Optional[str] = Field(
        None, description="low | medium | high"
    )
    month: Optional[str] = Field(
        None, description="Month of travel"
    )
    budget: Optional[str] = Field(
        None, description="low | medium | high | not specified"
    )
    interests: Optional[str] = Field(
        None, description="Comma-separated or free-text interests"
    )
    aqi: Optional[str] = Field(
        None, description="great | moderate | good | not specified | unhealthy | unhealthy for some"
    )
    walkability: Optional[str] = Field(
        None, description="great | okay | bad | not specified"
    )
    seasonality: Optional[str] = Field(
        None, description="low | medium | high"
    )

    @field_validator("seasonality", "budget", "popularity")
    @classmethod
    def filter_must_be_valid(cls, v):
        if v.lower() not in ["low", "medium", "high", "great"]:
            raise ValueError(f"Filter '{v}' is not valid.")
        return v

    @field_validator("aqi", "walkability")
    @classmethod
    def filter_must_be_valid(cls, v):
        if v.lower() not in ["great", "moderate", "good", "not specified", "unhealthy", "unhealthy for some", "okay",
                             "bad"]:
            raise ValueError(f"Filter '{v}' is not valid.")
        return v

    @field_validator("interests")
    @classmethod
    def interests_must_be_valid(cls, v):
        if v.lower() not in ["arts & entertainment",
                             "outdoors & recreation",
                             "food", "nightlife spots",
                             "shops & services", "nightlife spot"]:
            raise ValueError(f"Interest '{v}' is not valid.")
        return v


class AgentRequest(BaseModel):
    config_id: Optional[str] = None
    query: str
    filters: UserFilters

    moderator_context: Optional[ModeratorContext] = Field(
        None,
        description="Negotiation context provided by the moderator in rounds > 0"
    )