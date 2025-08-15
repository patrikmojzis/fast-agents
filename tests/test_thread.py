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
async def test_simple_thread_creation():
    """Test simple thread creation without complex setup."""
    from fast_agents import Agent, Thread
    
    agent = Agent(name="simple", instructions="Simple test agent")
    thread = Thread(agent=agent, input=[], max_turns=1, model="gpt-4")
    
    assert thread.agent.name == "simple"
    assert thread.max_turns == 1
    assert thread.input == []