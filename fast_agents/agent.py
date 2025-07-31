from __future__ import annotations

from typing import Optional, Type

from pydantic import BaseModel, Field, ConfigDict

from fast_agents.tool import Tool


class Agent(BaseModel):
    name: str = Field("agent")
    instructions: str = Field("You are a helpful assistant.")
    temperature: Optional[float] = Field(None, description="Value between 0 and 2. Reasoning models don't support temperature.")
    tools: list[Tool] = Field([])
    output_type: Optional[Type[BaseModel]] = Field(None)
    model: Optional[str] = Field(None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )
