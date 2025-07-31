from abc import abstractmethod, ABC

from pydantic import BaseModel, Field


class LlmContext(ABC, BaseModel):
    name: str = Field(...)

    @abstractmethod
    async def get_content(self) -> str:
        raise NotImplementedError

    async def dumps(self) -> str:
        return f"**{self.name}:**\n```{await self.get_content()}```"