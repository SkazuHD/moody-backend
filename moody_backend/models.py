from typing import Literal, List

from pydantic import BaseModel, constr, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: constr(min_length=1)


class ShortTermState(BaseModel):
    state: str = Field(..., description="A time-sensitive feeling or event.")
    last_mentioned: int = Field(..., description="Number of diary entries since last mention.")


class ContextualInsight(BaseModel):
    insight: str = Field(..., description="A person, event, or topic frequently mentioned.")
    last_mentioned: int = Field(..., description="Number of diary entries since last mention.")


class Persona(BaseModel):
    long_term_traits: List[str] = Field(..., description="Stable personality traits, values, and preferences.")
    short_term_states: List[ShortTermState] = Field(..., description="Recent emotional or situational states.")
    contextual_insights: List[ContextualInsight] = Field(...,
                                                         description="Commonly referenced people, themes, or objects.")


class AnalyzeResponse(BaseModel):
    mood: Literal["happy", "sad", "calm", "fearful", "angry", "disgust", "neutral", "surprised"]
    transcription: str = Field(..., description="Transcription of the diary entry.")
    recommendations: List[str] = Field(..., description="Suggestions for improving mood or well-being.")
    quote: constr(min_length=1) = Field(..., description="A random quote related to the mood.")
    personality: Persona = Field(...,
                                 description="Updated Personality traits and insights derived from the diary entry.")

class AnalyzeResponseFastCheckin(BaseModel):
    mood: Literal["happy", "sad", "calm", "fearful", "angry", "disgust", "neutral", "surprised"]
    recommendations: List[str] = Field(..., description="Suggestions for improving mood or well-being.")
    quote: constr(min_length=1) = Field(..., description="A random quote related to the mood.")