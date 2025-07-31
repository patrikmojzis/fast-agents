from abc import abstractmethod, ABC


class ToolTransformer(ABC):

    @abstractmethod
    async def transform(self, **kwargs) -> dict:
        """Transforms kwargs from function call"""
        raise NotImplementedError