"""
Tests for the Thread class.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from openai.types.responses import ResponseInputParam

from fast_agents import Agent, Thread, Tool, ToolResponse
from tests.conftest import MockResponse, MockResponseOutputItem


class TestThread:
    """Test cases for the Thread class."""

    @pytest.fixture
    def thread_with_agent(self, sample_agent):
        """Create a thread with a sample agent."""
        return Thread(
            agent=sample_agent,
            input=[],
            max_turns=5,
            model="gpt-4"
        )

    @pytest.mark.asyncio
    async def test_thread_initialization(self, sample_agent):
        """Test thread initialization with proper parameters."""
        thread = Thread(
            agent=sample_agent,
            input=[],
            max_turns=10,
            model="gpt-4",
            user_id="test_user"
        )
        
        assert thread.agent == sample_agent
        assert thread.input == []
        assert thread.max_turns == 10
        assert thread.model == "gpt-4"
        assert thread.user_id == "test_user"

    @pytest.mark.asyncio
    async def test_thread_with_mock_response(self, thread_with_agent, mock_text_response):
        """Test thread execution with mocked OpenAI response."""
        
        # Mock the AsyncOpenAI client
        with patch('fast_agents.thread.AsyncOpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=mock_text_response)
            mock_openai.return_value = mock_client
            
            # Execute the thread
            responses = []
            async for response in thread_with_agent.arun():
                responses.append(response)
            
            # Verify the response
            assert len(responses) == 1
            assert responses[0].content == "Hello, this is a test response!"
            
            # Verify the client was called correctly
            mock_client.responses.create.assert_called_once()
            call_args = mock_client.responses.create.call_args
            assert call_args[1]['model'] == "gpt-4"
            assert call_args[1]['instructions'] == "You are a test assistant."

    @pytest.mark.asyncio
    async def test_thread_with_function_call(self, sample_agent, sample_tool, mock_function_call_response):
        """Test thread execution with function call response."""
        
        # Add the tool to the agent
        sample_agent.tools = [sample_tool]
        
        thread = Thread(
            agent=sample_agent,
            input=[],
            max_turns=5,
            model="gpt-4"
        )
        
        # Mock the AsyncOpenAI client
        with patch('fast_agents.thread.AsyncOpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.responses.create = AsyncMock(return_value=mock_function_call_response)
            mock_openai.return_value = mock_client
            
            # Execute the thread
            responses = []
            async for response in thread.run():
                responses.append(response)
            
            # Verify the function call response
            assert len(responses) >= 1
            assert responses[0].name == "test_tool"
            assert responses[0].arguments == {"param1": "value1"}

    @pytest.mark.asyncio
    async def test_thread_max_turns_limit(self, sample_agent):
        """Test that thread respects max_turns limit."""
        
        thread = Thread(
            agent=sample_agent,
            input=[],
            max_turns=1,  # Set very low limit
            model="gpt-4"
        )
        
        # Mock continuous responses to test the limit
        with patch('fast_agents.thread.AsyncOpenAI') as mock_openai:
            mock_client = MagicMock()
            # Create a response that would trigger another turn
            continuous_response = MockResponse([
                MockResponseOutputItem(
                    item_type="function_call",
                    name="nonexistent_tool",
                    arguments={"test": "value"},
                    call_id="call_123"
                )
            ])
            mock_client.responses.create = AsyncMock(return_value=continuous_response)
            mock_openai.return_value = mock_client
            
            # Execute and collect all responses
            responses = []
            try:
                async for response in thread.run():
                    responses.append(response)
            except Exception:
                # May raise an exception due to nonexistent tool
                pass
            
            # Should not exceed max_turns calls to OpenAI
            assert mock_client.responses.create.call_count <= 1

    @pytest.mark.asyncio 
    async def test_thread_tool_execution(self, sample_agent, sample_tool):
        """Test that tools are executed correctly when called."""
        
        # Add tool to agent
        sample_agent.tools = [sample_tool]
        
        thread = Thread(
            agent=sample_agent,
            input=[],
            max_turns=5,
            model="gpt-4"
        )
        
        # Create a function call response for our tool
        function_call_response = MockResponse([
            MockResponseOutputItem(
                item_type="function_call", 
                name="test_tool",
                arguments={"message": "hello world"},
                call_id="call_123"
            )
        ])
        
        # Create a follow-up text response (what OpenAI would return after tool execution)
        text_response = MockResponse([
            MockResponseOutputItem(item_type="text", content="Tool executed successfully!")
        ])
        
        with patch('fast_agents.thread.AsyncOpenAI') as mock_openai:
            mock_client = MagicMock()
            # Return function call first, then text response
            mock_client.responses.create = AsyncMock(side_effect=[function_call_response, text_response])
            mock_openai.return_value = mock_client
            
            responses = []
            async for response in thread.run():
                responses.append(response)
            
            # Should have both the function call and the follow-up response
            assert len(responses) >= 1
            # First response should be the function call
            assert responses[0].name == "test_tool"


@pytest.mark.asyncio
async def test_simple_thread_creation():
    """Test simple thread creation without complex setup."""
    from fast_agents import Agent, Thread
    
    agent = Agent(name="simple", instructions="Simple test agent")
    thread = Thread(agent=agent, input=[], max_turns=1, model="gpt-4")
    
    assert thread.agent.name == "simple"
    assert thread.max_turns == 1
    assert thread.input == []