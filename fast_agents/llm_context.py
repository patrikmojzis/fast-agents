from abc import abstractmethod, ABC
from typing import ClassVar, Optional


class LlmContext(ABC):
    # Static metadata configured on subclasses; defaulted via __init_subclass__
    name: ClassVar[Optional[str]] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Default name to class name if not provided or falsy
        if not getattr(cls, "name", None):
            cls.name = cls.__name__

    @abstractmethod
    async def get_content(self) -> str:
        raise NotImplementedError

    async def dumps(self) -> str:
        return f"**{self.name}:**\n```{await self.get_content()}```"