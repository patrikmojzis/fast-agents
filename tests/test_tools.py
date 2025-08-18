"""
Tests for the Tool classes.
"""

import pytest
from pydantic import BaseModel

from fast_agents import Tool, ToolResponse


class TestTool:
    """Test cases for the Tool base class."""

    def test_tool_definition_generation(self):
        """Test that tools generate proper tool definitions."""
        
        class TestToolSchema(BaseModel):
            message: str
            count: int = 1
        
        class TestTool(Tool):
            """A tool for testing"""
            name = "test_tool"
            schema = TestToolSchema
            
            async def handle(self, **kwargs) -> ToolResponse:
                return ToolResponse(output="test")
        
        tool = TestTool()
        definition = tool.tool_definition
        
        assert definition["type"] == "function"
        assert definition["name"] == "test_tool"
        assert definition["description"] == "A tool for testing"
        assert "parameters" in definition
        assert definition["parameters"]["type"] == "object"
        assert "message" in definition["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_tool_handle_method(self):
        """Test tool handle method execution."""
        
        class EchoTool(Tool):
            name = "echo_tool"
            description = "Echoes input"
            
            async def handle(self, message: str = "default", **kwargs) -> ToolResponse:
                return ToolResponse(output=f"Echo: {message}")
        
        tool = EchoTool()
        response = await tool.handle(message="hello world")
        
        assert isinstance(response, ToolResponse)
        assert response.output == {"message": "Echo: hello world"}
        assert response.is_error is False

    @pytest.mark.asyncio
    async def test_tool_error_response(self):
        """Test tool error handling."""
        
        class ErrorTool(Tool):
            name = "error_tool"
            description = "Always errors"
            
            async def handle(self, **kwargs) -> ToolResponse:
                return ToolResponse(output="Something went wrong", is_error=True)
        
        tool = ErrorTool()
        response = await tool.handle()
        
        assert response.is_error is True
        assert "[Error]" in response.output_str

    


class TestToolResponse:
    """Test cases for ToolResponse."""

    def test_tool_response_basic(self):
        """Test basic tool response creation."""
        
        response = ToolResponse(output="test message")
        
        assert response.output == {"message": "test message"}
        assert response.is_error is False
        assert response.additional_inputs is None

    def test_tool_response_dict_output(self):
        """Test tool response with dict output."""
        
        output_data = {"result": "success", "count": 5}
        response = ToolResponse(output=output_data)
        
        assert response.output == output_data
        assert response.is_error is False

    def test_tool_response_error(self):
        """Test error tool response."""
        
        response = ToolResponse(output="Error occurred", is_error=True)
        
        assert response.is_error is True
        assert "[Error]" in response.output_str

    def test_tool_response_output_str(self):
        """Test output string generation."""
        
        response = ToolResponse(output="test")
        output_str = response.output_str
        
        assert "test" in output_str
        assert "[Error]" not in output_str

