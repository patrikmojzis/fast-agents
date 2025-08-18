from __future__ import annotations

from typing import Optional, Type

from pydantic import BaseModel, Field, ConfigDict

from fast_agents.tool import Tool


class Agent(BaseModel):
    name: str = Field("agent")
    instructions: str = Field("You are a helpful assistant.")
    temperature: Optional[float] = Field(None, description="Value between 0 and 2. Reasoning models don't support temperature.")
    tools: list[Tool] = Field([], description="List of tools available.")
    output_type: Optional[Type[BaseModel]] = Field(None, description="Structured output.")
    model: Optional[str] = Field(None, description="Default model to use. Can be overridden in the thread.")

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )
