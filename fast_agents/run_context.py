from typing import Optional, Any, TYPE_CHECKING

from openai.types.responses import ResponseInputParam
from pydantic import BaseModel, Field, ConfigDict

from fast_agents.agent import Agent

if TYPE_CHECKING:
    from fast_agents.thread import Thread


class RunContext(BaseModel):
    agent: Agent = Field(..., description="The agent that is running the thread.")
    turn: int = Field(..., description="The turn number of the thread.")
    max_turns: int = Field(..., description="The maximum number of turns the thread can take.")
    input: ResponseInputParam | list[Any] = Field(..., description="The input of the thread.")
    context: Optional['BaseModel'] = Field(None, description="The context of the thread.")
    # thread: 'Thread' = Field(..., description="The thread that is running.")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )


