from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import pydantic
from pydantic import BaseModel

from fast_agents.exceptions import ToolValidationException
from fast_agents.helpers.schema_helper import format_parameters
from fast_agents.tool_response import ToolResponse
from fast_agents.tool_transformer import ToolTransformer
from fast_agents.tool_transformers.id_to_object_id_transformer import IdToObjectIdTransformer
from fast_agents.tool_transformers.json_extractor_transformer import JsonExtractorTransformer
from fast_agents.tool_transformers.string_to_date_transformer import \
    StringToDateTransformer
from fast_agents.tool_validator import ToolValidator

if TYPE_CHECKING:
    from fast_agents.run_context import RunContext

class Tool(ABC):
    name: str = None
    description: str = None
    tool_schema: type[BaseModel] = None

    validators: list[type[ToolValidator]] = []
    transformers: list[type[ToolTransformer]] = []
    shared_transformers: list[type[ToolTransformer]] = [
        JsonExtractorTransformer([]),
        IdToObjectIdTransformer([]),
        StringToDateTransformer([])
    ]
    dump_config: Optional[dict] = {}

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
            response_dict = response.model_dump(**self.dump_config)
        except pydantic.ValidationError as e:
            return ToolResponse(to_agent=str(e), is_error=True)

        # Transform
        for transformer in self.shared_transformers + self.transformers:
            response_dict = await transformer.transform(**response_dict)

        # Validate
        for validator in self.validators:
            try:
                await validator.validate(**kwargs)
            except ToolValidationException as e:
                return ToolResponse(to_agent=str(e), is_error=True)

        # Execute
        return await self.handle(**response_dict)
