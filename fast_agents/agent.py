from __future__ import annotations

from typing import Optional, Type

from openai.types import ReasoningEffort
from pydantic import BaseModel, Field, ConfigDict

from fast_agents.helpers.handoffs_helper import HandoffTool
from fast_agents.tool import Tool


class Agent(BaseModel):
    name: str = Field("agent")
    instructions: str = Field("You are a helpful assistant.")
    model: str = Field("gpt-5.1", description="Default model to use.")
    tools: list[Tool | dict] = Field([], description="List of tools available. Can be a list of Tool objects or a list of native tool like code_interpreter, web_search_preview, image_generation.")
    output_type: Optional[Type[BaseModel]] = Field(None, description="Structured output.")
    temperature: Optional[float] = Field(None, description="Value between 0 and 2. Reasoning models don't support temperature.")
    reasoning_effort: Optional[ReasoningEffort] = Field(None, description="Reasoning effort for reasoning-capable models.")

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    def as_handoff_tool(self) -> HandoffTool:
        tool = HandoffTool()
        tool.name = tool.name.replace("<agent_name>", self.name)
        tool.description = tool.description.replace("<agent_name>", self.name)
        tool.agent = self
        return tool