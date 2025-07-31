"""
Tests for the Tool classes and tool transformers.
"""

import pytest
from pydantic import BaseModel

from fast_agents import Tool, ToolResponse, ToolTransformer, ToolValidator
from fast_agents import IdToObjectIdTransformer, JsonExtractorTransformer
from bson import ObjectId


class TestTool:
    """Test cases for the Tool base class."""

    def test_tool_definition_generation(self):
        """Test that tools generate proper tool definitions."""
        
        class TestToolSchema(BaseModel):
            message: str
            count: int = 1
        
        class TestTool(Tool):
            name = "test_tool"
            description = "A tool for testing"
            tool_schema = TestToolSchema
            
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


class TestToolTransformers:
    """Test cases for tool transformers."""

    @pytest.mark.asyncio
    async def test_id_to_object_id_transformer(self):
        """Test ObjectId transformation."""
        
        transformer = IdToObjectIdTransformer(["user_id"])
        
        # Test valid ObjectId string
        valid_id = str(ObjectId())
        result = await transformer.transform(user_id=valid_id)
        
        assert isinstance(result["user_id"], ObjectId)
        assert str(result["user_id"]) == valid_id

    @pytest.mark.asyncio 
    async def test_id_to_object_id_transformer_invalid(self):
        """Test ObjectId transformation with invalid ID."""
        
        transformer = IdToObjectIdTransformer(["user_id"])
        
        # Test invalid ObjectId string
        result = await transformer.transform(user_id="invalid_id")
        
        # Should not transform invalid IDs
        assert result["user_id"] == "invalid_id"

    @pytest.mark.asyncio
    async def test_json_extractor_transformer(self):
        """Test JSON extraction from strings."""
        
        transformer = JsonExtractorTransformer(["data"])
        
        # Test valid JSON string
        json_string = '{"key": "value", "number": 42}'
        result = await transformer.transform(data=json_string)
        
        assert isinstance(result["data"], dict)
        assert result["data"]["key"] == "value"
        assert result["data"]["number"] == 42

    @pytest.mark.asyncio
    async def test_json_extractor_transformer_invalid(self):
        """Test JSON extraction with invalid JSON."""
        
        transformer = JsonExtractorTransformer(["data"])
        
        # Test invalid JSON string
        result = await transformer.transform(data="not json")
        
        # Should not transform invalid JSON
        assert result["data"] == "not json"


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

    def test_tool_response_with_object_id(self):
        """Test tool response with ObjectId serialization."""
        
        obj_id = ObjectId()
        response = ToolResponse(output={"id": obj_id})
        
        # Check that the ObjectId is in the output dict
        assert response.output["id"] == obj_id
        
        # The model_dump should handle ObjectId serialization via json_encoders
        model_data = response.model_dump()
        assert str(obj_id) in str(model_data)