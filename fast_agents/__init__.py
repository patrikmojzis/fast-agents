"""
Fast Agents - A clean, standalone AI framework package for building AI agents.

This package provides a simple and elegant framework for building AI agents
with OpenAI's API, featuring tool-based extensibility, built-in validation
and transformation, async/await support, and type-safe Pydantic models.
"""

from fast_agents.agent import Agent
from fast_agents.tool import Tool
from fast_agents.tool_response import ToolResponse
from fast_validation import Schema, ValidatorRule, ValidationRuleException
from fast_agents.thread import Thread
from fast_agents.run_context import RunContext
from fast_agents.llm_context import LlmContext
from fast_agents.hook import Hook
from fast_agents.exceptions import (
    ToolValidationException,
    MaxTurnsReachedException,
    RefusalException,
    InvalidJSONResponseException,
    InvalidPydanticSchemaResponseException,
)


__version__ = "0.1.0"
__author__ = "Patrik Mojzis"
__email__ = "patrikm53@gmail.com"

# Define what gets imported with "from fast_agents import *"
__all__ = [
    # Core classes
    "Agent",
    "Tool", 
    "ToolResponse",
    "Schema",
    "ValidatorRule",
    "ValidationRuleException",
    "Thread",
    "RunContext",
    "LlmContext",
    "Hook",

    # Exceptions
    "ToolValidationException",
    "MaxTurnsReachedException", 
    "RefusalException",
    "InvalidJSONResponseException",
    "InvalidPydanticSchemaResponseException",
    
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
]