from typing import Literal

from pydantic import BaseModel, constr


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: constr(min_length=1)

class AnalyzeResponse(BaseModel):
    mood: Literal["happy","sad","calm","fearful","angry", "disgust", "neutral", "suprised"]
    recommendations: list
    quote: constr(min_length=1)