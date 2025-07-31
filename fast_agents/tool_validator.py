from abc import abstractmethod, ABC

class ToolValidator(ABC):

    @abstractmethod
    async def validate(self, **kwargs):
        """
        If validation fails raise ToolValidationException with message to llm.

        :param kwargs: function call args
        :return:
        """
        raise NotImplementedError
