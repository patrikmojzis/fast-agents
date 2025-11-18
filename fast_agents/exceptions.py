class ToolValidationException(Exception):
    pass


class AgentException(Exception):
    pass


class MaxTurnsReachedException(AgentException):
    def __init__(self):
        super().__init__("Max turns reached")


class RefusalException(AgentException):
    pass


class InvalidJSONResponseException(AgentException):
    pass


class InvalidPydanticSchemaResponseException(AgentException):
    pass


class StreamingFailedException(AgentException):
    def __init__(self, message: str = "Streaming failed"):
        super().__init__(message)


class ConfigurationException(ValueError):
    pass


class ToolException(Exception):
    """
    Raise this exception inside a Tool to propagate to Agent an error.
    This exception does not cause crashes. It is caught in subsequent steps
    and its message is passed to agent.
    """
    pass
