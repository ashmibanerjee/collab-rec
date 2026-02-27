from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from constants import CITIES


class ModeratorContext(BaseModel):
    round: int = Field(
        ..., description="Current negotiation round (starting from 1)"
    )

    collective_offer: Optional[List[str]] = Field(
        None,
        description="Top-k highest-scoring cities from the previous round (Current Collective Offer ϕ_t). You must keep at least 7 of these cities."
    )

    additional_candidates: Optional[List[str]] = Field(
        None,
        description="Additional available cities from the catalog (not in collective_offer and not rejected). You can use up to 3 of these to replace cities from the collective offer."
    )

    collective_rejection: Optional[List[str]] = Field(
        None,
        description="Cities rejected in previous rounds. Do NOT recommend any of these cities."
    )

    previous_recommendations: Optional[List[str]] = Field(
        None,
        description="Your own candidates list from the previous round"
    )

    agent_feedback: Optional[str] = Field(
        None,
        description="Feedback from moderator: CONSTRAINT - You must keep at least 7 cities from collective_offer and can replace at most 3 with cities from additional_candidates."
    )

