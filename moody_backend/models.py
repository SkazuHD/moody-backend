from typing import Literal

from pydantic import BaseModel, constr


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: constr(min_length=1)
