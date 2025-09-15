"""
Tests for handoffs via RunPipeline postflight integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fast_agents import Agent, Thread
from fast_agents.helpers.handoffs_helper import HandoffRunPipeline
from tests.conftest import MockResponse, MockResponseOutputItem


@pytest.mark.asyncio
async def test_handoff_pipeline_switches_agent():
    """When a function_call targets a handoff tool, the pipeline should switch the thread.agent."""

    # Create main agent and target agent
    main_agent = Agent(name="main", instructions="Main agent", model="gpt-4")
    target_agent = Agent(name="target", instructions="Target agent", model="gpt-4")

    # Expose a handoff tool on the main agent that points to the target agent
    handoff_tool = target_agent.as_handoff_tool()
    main_agent.tools = [handoff_tool]

    # Thread with the handoff pipeline enabled
    thread = Thread(
        agent=main_agent,
        input=[],
        max_turns=2,
        run_pipelines=[HandoffRunPipeline()],
    )

    # Mock OpenAI response: one function_call to the handoff tool
    mock_response = MockResponse([
        MockResponseOutputItem(
            item_type="function_call",
            name=handoff_tool.name,
            arguments='{"reason": "Escalate to target"}',
            call_id="call_1",
        )
    ])

    # Patch the OpenAI client used inside Thread
    with patch("fast_agents.thread.AsyncOpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_second_response = MockResponse([
            MockResponseOutputItem(item_type="text", content="done")
        ])
        mock_client.responses.create = AsyncMock(side_effect=[mock_response, mock_second_response])
        mock_openai_class.return_value = mock_client

        # Run the thread once to process the response and pipeline postflight
        outputs = []
        async for item in thread.run():
            outputs.append(item)

    # Agent should have switched to target agent
    assert thread.agent is target_agent
    assert thread.agent.name == "target"


