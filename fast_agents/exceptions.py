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


class ValidationRuleException(ToolValidationException):
    def __init__(
        self,
        message: str,
        *,
        loc: tuple[str, ...] | tuple = tuple(),
        error_type: str = "rule_error",
        errors: list[dict] | None = None,
    ) -> None:
        super().__init__(message)
        self.loc = loc
        self.error_type = error_type
        self.errors = errors or []