"""
Tests for helper functions.
"""

import pytest
from fast_agents.helpers import format_parameters, num_tokens_from_string, gather_contexts
from fast_agents import LlmContext
from pydantic import BaseModel
from typing import Optional


class TestHelpers:
    """Test cases for helper functions."""

    def test_num_tokens_from_string(self):
        """Test token counting functionality."""
        
        # Test basic string
        tokens = num_tokens_from_string("Hello world")
        assert isinstance(tokens, int)
        assert tokens > 0
        
        # Test empty string
        empty_tokens = num_tokens_from_string("")
        assert empty_tokens == 0
        
        # Test longer text
        long_text = "This is a longer piece of text that should have more tokens than the simple hello world example."
        long_tokens = num_tokens_from_string(long_text)
        assert long_tokens > tokens

    def test_num_tokens_with_different_encodings(self):
        """Test token counting with different encodings."""
        
        text = "Hello world"
        
        # Test default encoding
        default_tokens = num_tokens_from_string(text)
        
        # Test specific encoding
        specific_tokens = num_tokens_from_string(text, encoding_name="o200k_base")
        
        assert default_tokens == specific_tokens
        assert isinstance(default_tokens, int)

    def test_format_parameters(self):
        """Test parameter formatting for tool schemas."""
        
        class TestSchema(BaseModel):
            name: str
            age: int = 25
            email: Optional[str] = None
            active: bool = True
        
        formatted = format_parameters(TestSchema)
        
        assert formatted["type"] == "object"
        assert "properties" in formatted
        assert "name" in formatted["properties"]
        assert "age" in formatted["properties"]
        assert "email" in formatted["properties"]
        assert "active" in formatted["properties"]
        
        # Check required fields
        assert "required" in formatted
        assert "name" in formatted["required"]  # Required field
        assert "email" not in formatted["required"]  # Optional field

    def test_format_parameters_with_complex_types(self):
        """Test parameter formatting with complex types."""
        
        class NestedSchema(BaseModel):
            value: str
        
        class ComplexSchema(BaseModel):
            simple_field: str
            nested_field: NestedSchema
            list_field: list[str] = []
        
        formatted = format_parameters(ComplexSchema)
        
        assert formatted["type"] == "object"
        assert "simple_field" in formatted["properties"]
        assert "nested_field" in formatted["properties"]
        assert "list_field" in formatted["properties"]

    @pytest.mark.asyncio
    async def test_gather_contexts(self):
        """Test LLM context gathering."""
        
        class MockLlmContext(LlmContext):
            model_config = {"extra": "allow"}  # Allow extra fields
            
            def __init__(self, content: str):
                super().__init__(name="test_context")
                self.content = content
            
            async def dumps(self) -> str:
                return self.content
                
            async def get_content(self) -> str:
                return self.content
        
        # Create test contexts
        context1 = MockLlmContext("Context 1 content")
        context2 = MockLlmContext("Context 2 content")
        contexts = [context1, context2]
        
        result = await gather_contexts(contexts)
        
        assert isinstance(result, str)
        assert "Context 1 content" in result
        assert "Context 2 content" in result
        assert "\n\n" in result  # Should be joined with double newlines

    @pytest.mark.asyncio 
    async def test_gather_contexts_empty(self):
        """Test gathering empty context list."""
        
        result = await gather_contexts([])
        
        assert result == ""

    @pytest.mark.asyncio 
    async def test_gather_contexts_single(self):
        """Test gathering single context."""
        
        class MockLlmContext(LlmContext):
            def __init__(self):
                super().__init__(name="test_context")
                
            async def dumps(self) -> str:
                return "Single context"
                
            async def get_content(self) -> str:
                return "Single context"
        
        context = MockLlmContext()
        result = await gather_contexts([context])
        
        assert result == "Single context"