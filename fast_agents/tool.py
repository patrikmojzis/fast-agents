from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Optional

import pydantic
from fast_validation import ValidationRuleException, Schema
from pydantic import BaseModel

from fast_agents.exceptions import ConfigurationException, ToolException
from fast_agents.helpers.schema_helper import format_parameters
from fast_agents.helpers.text_helper import pascal_case_to_snake_case
from fast_agents.tool_response import ToolResponse

if TYPE_CHECKING:
    from fast_agents.run_context import RunContext

class Tool(ABC):
    # Static metadata configured on subclasses
    name: ClassVar[Optional[str]] = None
    description: ClassVar[Optional[str]] = None
    schema: ClassVar[type[BaseModel] | type[Schema] | None] = None

    # Whether to treat inputs as partial updates (exclude unset fields)
    partial: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Default sensible metadata
        if not getattr(cls, "name", None):
            cls.name = pascal_case_to_snake_case(cls.__name__).rstrip("_tool")
        # Derive description from docstring if not explicitly provided
        if not getattr(cls, "description", None):
            doc = cls.__doc__
            if isinstance(doc, str):
                first_line = doc.strip().splitlines()[0].strip() if doc.strip() else None
                cls.description = first_line or None
            else:
                cls.description = None

    def __init__(self) -> None:
        self.run_context: 'RunContext' = None

        if not self.schema:
            raise ConfigurationException("Tool schema is not defined. Define a Pydantic BaseModel in `schema`.\nExample:\n\nclass MyTool(Tool):\n    schema = MyToolSchema")
        
        if not self.name:
            raise ConfigurationException("Tool name is not defined. Define a name in `name`.\nExample:\n\nclass MyTool(Tool):\n    name = 'my_tool'")
        
        if not self.description:
            raise ConfigurationException("Tool description is not defined. Define a description in `description`.\nExample:\n\nclass MyTool(Tool):\n    description = 'My tool description'")
        
    @property
    def tool_definition(self) -> dict:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": format_parameters(self.schema)
        }

    @abstractmethod
    async def handle(self, **kwargs) -> ToolResponse | dict | str | int | list:
        raise NotImplementedError

    async def arun(self, run_context: 'RunContext', **kwargs) -> ToolResponse:
        self.run_context = run_context

        # Parse
        try:
            response = self.schema(**kwargs)
            response_dict = response.model_dump(exclude_unset=self.partial)
        except pydantic.ValidationError as e:
            return ToolResponse(is_error=True, output=str(e))

        # Schema rule validation (optional)
        if isinstance(response, Schema):
            try:
                await response.validate(partial=self.partial)
            except ValidationRuleException as e:
                return ToolResponse(is_error=True, output=e.errors)

        # Execute
        try:
            res = await self.handle(**response_dict)
            if isinstance(res, ToolResponse):
                return res

            return ToolResponse(output=res)
        except ToolException as e:
            return ToolResponse(is_error=True, output=str(e))
