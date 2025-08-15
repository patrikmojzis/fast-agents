from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pydantic
from pydantic import BaseModel

from fast_agents.exceptions import ValidationRuleException
from fast_agents.helpers.schema_helper import format_parameters
from fast_agents.tool_response import ToolResponse
from fast_agents.schema import Schema

if TYPE_CHECKING:
    from fast_agents.run_context import RunContext

class Tool(ABC):
    name: str = None
    description: str = None
    
    tool_schema: type[BaseModel] | type[Schema] = None   # Accept any Pydantic model class, including our Schema subclasses

    partial: bool = False   # Whether to treat inputs as partial updates (exclude unset fields)

    def __init__(self) -> None:
        self.run_context: 'RunContext' = None

    @property
    def tool_definition(self) -> dict:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": format_parameters(self.tool_schema)
        }

    @abstractmethod
    async def handle(self, **kwargs) -> ToolResponse:
        raise NotImplementedError

    async def arun(self, run_context: 'RunContext', **kwargs) -> ToolResponse:
        self.run_context = run_context

        # Parse
        try:
            response = self.tool_schema(**kwargs)
            response_dict = response.model_dump(exclude_unset=self.partial)
        except pydantic.ValidationError as e:
            return ToolResponse(output=str(e), is_error=True)

        # Schema rule validation (optional)
        if isinstance(response, Schema):
            try:
                await response.validate(partial=self.partial)
            except ValidationRuleException as e:
                return ToolResponse(output=str(e), is_error=True)

        # Execute
        return await self.handle(**response_dict)
