from typing import Optional, Any

from openai.types.responses import ResponseInputParam
from pydantic import BaseModel, Field

from fast_agents.agent import Agent


class RunContext(BaseModel):
    agent: Agent = Field(..., description="The agent that is running the thread.")
    turn: int = Field(..., description="The turn number of the thread.")
    max_turns: int = Field(..., description="The maximum number of turns the thread can take.")
    model: str = Field(..., description="The model that is running the thread.")
    input: ResponseInputParam | list[Any] = Field(..., description="The input of the thread.")
    context: Optional['BaseModel'] = Field(None, description="The context of the thread.")
