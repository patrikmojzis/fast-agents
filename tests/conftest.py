"""
Test configuration and fixtures for fast-agents tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from openai.types.responses import ResponseOutputItem
from typing import Any, Dict, List


class MockResponseOutputItem:
    """Mock ResponseOutputItem for testing."""
    
    def __init__(self, item_type: str = "text", content: str = "test content", 
                 name: str = None, arguments: Dict[str, Any] = None, call_id: str = None):
        self.type = item_type
        self.content = content
        self.name = name
        self.arguments = arguments or {}
        self.call_id = call_id


class MockResponse:
    """Mock OpenAI response for testing."""
    
    def __init__(self, output: List[MockResponseOutputItem] = None):
        self.output = output or [MockResponseOutputItem()]


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client with async response creation."""
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock()
    return client


@pytest.fixture
def mock_text_response():
    """Create a mock text response."""
    return MockResponse([
        MockResponseOutputItem(item_type="text", content="Hello, this is a test response!")
    ])


@pytest.fixture
def mock_function_call_response():
    """Create a mock function call response."""
    return MockResponse([
        MockResponseOutputItem(
            item_type="function_call",
            name="test_tool",
            arguments={"param1": "value1"},
            call_id="call_123"
        )
    ])


@pytest.fixture
def sample_agent():
    """Create a sample agent for testing."""
    from fast_agents import Agent
    return Agent(
        name="test_agent",
        instructions="You are a test assistant.",
        tools=[],
        model="gpt-4"
    )


@pytest.fixture
def sample_tool():
    """Create a sample tool for testing."""
    from fast_agents import Tool, ToolResponse
    from pydantic import BaseModel
    
    class TestToolSchema(BaseModel):
        message: str
    
    class TestTool(Tool):
        name = "test_tool"
        description = "A test tool"
        tool_schema = TestToolSchema
        
        async def handle(self, **kwargs) -> ToolResponse:
            return ToolResponse(output=f"Handled: {kwargs.get('message', 'no message')}")
    
    return TestTool()