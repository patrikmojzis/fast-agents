from typing import TYPE_CHECKING, Optional
from fast_agents.run_pipeline import RunPipeline
from fast_agents.tool import Tool, ToolResponse
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fast_agents import Thread, Agent
    from openai.types.responses import Response


class HandoffToolSchema(BaseModel):
    reason: str = Field(..., description="Brief reason for handoff. Max 50 characters.")


class HandoffTool(Tool):
    name = "handoff_to_<agent_name>"
    description = "Proactively handoff the task to <agent_name> if relevant."
    schema = HandoffToolSchema
    agent: 'Agent' = None

    async def handle(self, **kwargs) -> ToolResponse:
        return ToolResponse(output_str="You have been handed this matter. Please, take over.")


class HandoffRunPipeline(RunPipeline):

    async def postflight(self, thread: 'Thread', response: 'Response'):
        # Find all function calls and handle those that target a HandoffTool
        function_calls = [
            output for output in response.output
            if getattr(output, "type", None) == "function_call"
        ]
        if not function_calls:
            return

        for fn_call in function_calls:
            matching_tool = next((tool for tool in thread.agent.tools if getattr(tool, "name", None) == getattr(fn_call, "name", None)), None)
            if matching_tool is None:
                continue

            # Only switch agent for handoff tools that have a target agent set
            if isinstance(matching_tool, HandoffTool) and getattr(matching_tool, "agent", None) is not None:
                thread.agent = matching_tool.agent
            
            