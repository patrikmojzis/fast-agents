"""
Tests for Thread class with proper mocking to avoid OpenAI API key requirement.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fast_agents import Agent, Thread, Tool, ToolResponse
from tests.conftest import MockResponse, MockResponseOutputItem


class TestThreadWithMocking:
    """Test Thread class with proper mocking."""

    @pytest.mark.asyncio
    async def test_thread_basic_creation(self):
        """Test that we can create a thread without API key issues."""
        
        agent = Agent(
            name="test_agent", 
            instructions="You are a test assistant.",
            model="gpt-4"
        )
        
        # Create thread without initializing OpenAI client
        thread = Thread(
            agent=agent,
            input=[],
            max_turns=5,
            model="gpt-4"
        )
        
        assert thread.agent == agent
        assert thread.max_turns == 5
        assert thread.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_thread_arun_with_mock(self):
        """Test thread.run() with mocked OpenAI client."""
        
        agent = Agent(
            name="test_agent",
            instructions="You are a test assistant.",
            model="gpt-4"
        )
        
        thread = Thread(
            agent=agent,
            input=[],
            max_turns=2,
            model="gpt-4"
        )
        
        # Create mock response
        mock_response = MockResponse([
            MockResponseOutputItem(item_type="text", content="Hello! How can I help you?")
        ])
        
        # Mock the AsyncOpenAI client creation and response
        with patch('fast_agents.thread.AsyncOpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=mock_response)
            mock_openai_class.return_value = mock_client
            
            # Execute the thread
            responses = []
            async for response in thread.run():
                responses.append(response)
            
            # Verify we got a response
            assert len(responses) == 1
            assert responses[0].content == "Hello! How can I help you?"
            
            # Verify the client was called
            mock_client.responses.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_thread_model_requirement(self):
        """Test that thread requires a model."""
        
        agent = Agent(
            name="test_agent",
            instructions="You are a test assistant."
            # No model specified
        )
        
        with pytest.raises(ValueError, match="Please provide a model"):
            Thread(
                agent=agent,
                input=[],
                max_turns=5
                # No model specified here either
            )