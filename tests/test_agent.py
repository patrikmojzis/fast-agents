"""
Tests for the Agent class.
"""

import pytest
from pydantic import ValidationError

from fast_agents import Agent, Tool, ToolResponse
from pydantic import BaseModel


class TestAgent:
    """Test cases for the Agent class."""

    def test_agent_creation_basic(self):
        """Test basic agent creation."""
        agent = Agent(
            name="test_agent",
            instructions="You are a helpful assistant."
        )
        
        assert agent.name == "test_agent"
        assert agent.instructions == "You are a helpful assistant."
        assert agent.tools == []
        assert agent.temperature is None

    def test_agent_creation_with_all_params(self):
        """Test agent creation with all parameters."""
        
        class TestTool(Tool):
            name = "test_tool"
            description = "A test tool"
            
            async def handle(self, **kwargs) -> ToolResponse:
                return ToolResponse(output="test")
        
        tool = TestTool()
        
        agent = Agent(
            name="full_agent",
            instructions="Detailed instructions",
            temperature=0.7,
            tools=[tool],
            model="gpt-4"
        )
        
        assert agent.name == "full_agent"
        assert agent.instructions == "Detailed instructions"
        assert agent.temperature == 0.7
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "test_tool"
        assert agent.model == "gpt-4"

    def test_agent_with_output_type(self):
        """Test agent with structured output type."""
        
        class OutputSchema(BaseModel):
            result: str
            confidence: float
        
        agent = Agent(
            name="structured_agent",
            instructions="Return structured data",
            output_type=OutputSchema
        )
        
        assert agent.output_type == OutputSchema

    def test_agent_validation(self):
        """Test agent field validation."""
        
        # Test that we can create agent with defaults - instructions has a default
        agent = Agent(name="test")  # Should work with default instructions
        assert agent.name == "test"
        
        # Test valid minimal creation
        agent = Agent(name="minimal", instructions="test")
        assert agent.name == "minimal"

    def test_agent_tools_list(self):
        """Test that agent tools is always a list."""
        
        agent = Agent(name="test", instructions="test")
        assert isinstance(agent.tools, list)
        assert len(agent.tools) == 0
        
        # Test with tools
        class DummyTool(Tool):
            name = "dummy"
            description = "dummy tool"
            
            async def handle(self, **kwargs) -> ToolResponse:
                return ToolResponse(output="dummy")
        
        tool1 = DummyTool()
        tool2 = DummyTool()
        
        agent_with_tools = Agent(
            name="test", 
            instructions="test",
            tools=[tool1, tool2]
        )
        
        assert len(agent_with_tools.tools) == 2