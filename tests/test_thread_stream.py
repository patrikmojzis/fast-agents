"""
Tests for Thread.stream streaming behavior with mocked OpenAI client.
"""

import pytest
from unittest.mock import MagicMock, patch

from fast_agents import Agent, Thread


class _AsyncStreamContext:
    def __init__(self, events, final_response):
        self._events = events
        self._final = final_response

    async def __aenter__(self):
        async def _aiter():
            for ev in self._events:
                yield ev
        self._aiter = _aiter()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        return self._aiter

    async def __anext__(self):
        return await self._aiter.__anext__()

    async def get_final_response(self):
        return self._final


class _MockResponseItem:
    def __init__(self, item_type="message", content_text="", name=None, arguments=None, call_id=None):
        self.type = item_type
        if item_type == "message":
            self.content = [{"type": "output_text", "text": content_text}]
        else:
            self.content = None
        self.name = name
        self.arguments = arguments
        self.call_id = call_id


class _MockFinalResponse:
    def __init__(self, outputs):
        self.output = outputs


@pytest.mark.asyncio
async def test_stream_text_events():
    agent = Agent(name="s", instructions="i", model="gpt-4o")
    thread = Thread(agent=agent, input=[{"role": "user", "content": "hi"}], max_turns=1)

    # Events: created is not needed by our code; we emit deltas then done
    events = [
        type("E", (), {"type": "response.output_text.delta", "output_index": 0, "delta": "Hel"})(),
        type("E", (), {"type": "response.output_text.delta", "output_index": 0, "delta": "lo"})(),
        type("E", (), {"type": "response.output_text.done", "output_index": 0, "text": "Hello"})(),
    ]
    final = _MockFinalResponse([_MockResponseItem(item_type="message", content_text="Hello")])

    with patch("fast_agents.thread.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.responses.stream.return_value = _AsyncStreamContext(events, final)
        mock_cls.return_value = mock_client

        outputs = []
        async for out in thread.stream():
            outputs.append(out)

    # We expect raw event objects then a final output object
    types = [o.get("type") if isinstance(o, dict) else getattr(o, "type", None) for o in outputs]
    assert types[:3] == [
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
    ]


@pytest.mark.asyncio
async def test_stream_function_call_events_and_tool_flow():
    # Provide a tool to satisfy call_tool
    from pydantic import BaseModel
    from fast_agents import Tool, ToolResponse

    class EchoSchema(BaseModel):
        message: str

    class EchoTool(Tool):
        name = "echo"
        description = "Echo tool"
        schema = EchoSchema
        async def handle(self, message: str, **kwargs) -> ToolResponse:
            return ToolResponse(output={"echo": message})


    agent = Agent(name="s", instructions="i", model="gpt-4o", tools=[EchoTool()])
    thread = Thread(agent=agent, input=[{"role": "user", "content": "call tool"}])

    # function call added -> args delta -> args done
    events = [
        type("E", (), {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": type("I", (), {"type": "function_call", "id": "fc1", "call_id": "call_1", "name": "echo"})(),
        })(),
        type("E", (), {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{\"m"})(),
        type("E", (), {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "essage\":\"hi\"}"})(),
        type("E", (), {"type": "response.function_call_arguments.done", "output_index": 0, "arguments": "{\"message\":\"hi\"}"})(),
    ]

    # Final response contains the same function call
    final = _MockFinalResponse([
        _MockResponseItem(item_type="function_call", name="echo", arguments="{\"message\":\"hi\"}", call_id="call_1")
    ])

    # Mock the OpenAI client: first stream call returns the function call,
    # second recursive stream call returns no events and a simple final message.
    with patch("fast_agents.thread.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.responses.stream.side_effect = [
            _AsyncStreamContext(events, final),
            _AsyncStreamContext([], _MockFinalResponse([
                _MockResponseItem(item_type="message", content_text="ok")
            ])),
        ]
        mock_cls.return_value = mock_client

        outputs = []
        async for out in thread.stream():
            outputs.append(out)

    # We expect raw event objects, the function_call output object, then tool response dict
    types = [o.get("type") if isinstance(o, dict) else getattr(o, "type", None) for o in outputs]
    assert "response.function_call_arguments.done" in types
    assert "function_call_output" in types
    # Ensure recursion happened exactly once (two stream calls total)
    assert mock_client.responses.stream.call_count == 2


