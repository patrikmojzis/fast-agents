class ToolValidationException(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class AgentException(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class MaxTurnsReachedException(AgentException):
    def __init__(self):
        super().__init__("Max turns reached")

class RefusalException(AgentException):
    def __init__(self, message: str):
        super().__init__(message)

class InvalidJSONResponseException(AgentException):
    def __init__(self, message: str):
        super().__init__(message)

class InvalidPydanticSchemaResponseException(AgentException):
    def __init__(self, message: str):
        super().__init__(message)


class StreamingFailedException(AgentException):
    def __init__(self, message: str = "Streaming failed"):
        super().__init__(message)


class ConfigurationException(ValueError):
    def __init__(self, message: str):
        super().__init__(message)