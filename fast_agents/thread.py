import asyncio
import json
from typing import TYPE_CHECKING, Optional, Callable, AsyncGenerator, Any

from openai import AsyncOpenAI
from openai.types import Reasoning
from openai.types.responses import ResponseInputParam, ResponseTextConfigParam, ResponseFormatTextJSONSchemaConfigParam, \
    ResponseOutputItem
from pydantic import ValidationError

from fast_agents.exceptions import MaxTurnsReachedException, RefusalException, InvalidJSONResponseException, \
    InvalidPydanticSchemaResponseException, StreamingFailedException
from fast_agents.helpers.input_filters import filter_ids, filter_status
from fast_agents.helpers.llm_context_helper import gather_contexts
from fast_agents.helpers.schema_helper import format_parameters
from fast_agents.helpers.tokenisor import num_tokens_from_string
from fast_agents.run_context import RunContext
from fast_agents.run_pipeline import RunPipeline
from fast_agents.tool_response import ToolResponse
from fast_agents import Tool

if TYPE_CHECKING:
    from fast_agents.agent import Agent
    from fast_agents.llm_context import LlmContext
    from fast_agents.hook import Hook
    from pydantic import BaseModel
    from openai.types.responses import Response
    
class Thread:
    def __init__(self,
                 agent: 'Agent',
                 input: Optional[ResponseInputParam] = None,
                 max_turns: int = 20,
                 context: Optional['BaseModel'] = None,
                 llm_contexts: Optional[list['LlmContext']] = None,
                 hooks: Optional[list['Hook']] = None,
                 max_input_tokens: Optional[int] = None,
                 prompt_cache_key: Optional[str] = None,
                 run_pipelines: Optional[list[RunPipeline]] = None,
                 openai_store_responses: Optional[bool] = True   # If True response objects are saved for 30 days. Opt out by setting to False. If using previous_response_id set True
                 ):
        self.agent = agent
        self.max_turns = max_turns
        self.input = (input or []).copy()   # copy to avoid mutating the original input
        self.context = context
        self.llm_contexts = llm_contexts
        self.hooks = hooks
        self.turn_count = 0 
        self.max_input_tokens = max_input_tokens
        self.prompt_cache_key = prompt_cache_key
        self.run_pipelines = run_pipelines
        self.client = None
        self.openai_store_responses = openai_store_responses
        
    def create_run_context(self, run_input: list[ResponseInputParam]) -> 'RunContext':
        return RunContext(
            agent=self.agent,
            turn=self.turn_count,
            max_turns=self.max_turns,
            input=run_input,
            context=self.context
        )

    def collect_function_calls(self, response: 'Response') -> list[tuple[str, str, str]]:
        return [
            (output.name, output.arguments, output.call_id)
            for output in response.output
            if output.type == "function_call"
        ]

    async def execute_tool_calls(self, function_calls: list[tuple[str, str, str]], run_context: 'RunContext', next_turn_coro: AsyncGenerator[ResponseOutputItem, Any]):
        tool_responses = await asyncio.gather(
            *[self.call_tool(name, args, run_context) for name, args, _ in function_calls])

        # Add tool responses to input
        for (name, args, call_id), response in zip(function_calls, tool_responses):
            fn_output = {"type": "function_call_output", "call_id": call_id, "output": response.output_str}
            self.input.append(fn_output)
            run_context.input.append(fn_output)
            yield fn_output
            if response.additional_inputs:
                for additional_input in response.additional_inputs:
                    self.input.append(additional_input)
                    run_context.input.append(additional_input)
                    yield additional_input

        # Recursively continue
        async for output in next_turn_coro():
            yield output

    async def get_run_input(self) -> list[ResponseInputParam]:     
        # TODO: refactor this to custom modular function and put into helpers like max_tokens max_messages etc.
        selected_inputs = []          

        if self.max_input_tokens: 
            remaining_tokens = self.max_input_tokens
            current_tokens = 0
            
            # Go through messages in reverse order (latest first)
            for msg in reversed(self.input):
                msg_tokens = num_tokens_from_string(str(msg))
                if current_tokens + msg_tokens <= remaining_tokens:
                    selected_inputs.insert(0, msg)  # Insert at beginning to maintain order
                    current_tokens += msg_tokens
                else:
                    break
        else:
            selected_inputs = self.input
        
        # Combine contexts (at start) with selected input messages
        contexts = await gather_contexts(self.llm_contexts) if self.llm_contexts else None

        if contexts:
            selected_inputs.insert(0, {"role": "system", "content": contexts})

        return filter_ids(filter_status(selected_inputs))

    def get_output_format(self) -> ResponseTextConfigParam:
        if output_type := self.agent.output_type:
            return ResponseTextConfigParam(
                format=ResponseFormatTextJSONSchemaConfigParam(
                    type="json_schema",
                    name=output_type.__name__,
                    description=output_type.__doc__,
                    schema=format_parameters(output_type),
                    strict=True
                )
            )

        return ResponseTextConfigParam(
            format={"type": "text"}
        )

    def tool_definitions(self) -> list[dict]:
        return [tool.tool_definition if isinstance(tool, Tool) else tool for tool in self.agent.tools]

    async def parse_structured_output(self, output: ResponseOutputItem) -> 'BaseModel':
        content = output.content[0]
        if content.type == "refusal":
            raise RefusalException(content.text)

        try:
            parsed_json = json.loads(content.text)
            return self.agent.output_type(**parsed_json)
        except json.JSONDecodeError as e:
            raise InvalidJSONResponseException(str(e))
        except ValidationError as e:
            raise InvalidPydanticSchemaResponseException(str(e))

    async def call_tool(self, name: str, args: str, run_context: 'RunContext') -> ToolResponse:
        for tool in self.agent.tools:
            if tool.name == name:
                try:
                    parsed_args = json.loads(args)
                except json.JSONDecodeError:
                    return ToolResponse(output=f"Invalid JSON: {args}", is_error=True)

                # Create a new instance of the tool for each call
                response = await tool.__class__().arun(**parsed_args, run_context=run_context)
                return response

        return ToolResponse(output=f"No tool found with name {name}", is_error=True)

    def verify_max_turns(self):
        if self.turn_count > self.max_turns:
            raise MaxTurnsReachedException()
        self.turn_count += 1

    async def run(self):
        self.verify_max_turns()

        if not self.client:
            self.client = AsyncOpenAI()

        if self.run_pipelines:
            await asyncio.gather(*[pipeline.preflight(self) for pipeline in self.run_pipelines])

        run_input = await self.get_run_input()

        run_context = self.create_run_context(run_input)

        if self.hooks:
            await asyncio.gather(*[hook.on_start(run_context) for hook in self.hooks])

        response = await self.client.responses.create(
            model=self.agent.model,
            instructions=self.agent.instructions,
            input=run_input,
            tools=self.tool_definitions(),
            temperature=self.agent.temperature,
            truncation="auto",
            text=self.get_output_format(),
            prompt_cache_key=self.prompt_cache_key,
            store=self.openai_store_responses, 
            reasoning=Reasoning(effort=self.agent.reasoning_effort) if getattr(self.agent, 'reasoning_effort', None) else None,
        )

        # Yield parts of the response
        for output in response.output:
            yield output

        self.input.extend(response.output)
        run_context.input.extend(response.output)

        if self.hooks:
            await asyncio.gather(*[hook.on_end(run_context, response) for hook in self.hooks])

        if self.run_pipelines:
            await asyncio.gather(*[pipeline.postflight(self, response) for pipeline in self.run_pipelines])

        # Execute all function calls 
        if function_calls := self.collect_function_calls(response):
            async for output in self.execute_tool_calls(function_calls, run_context, next_turn_coro=self.run):
                yield output

    async def stream(self):
        """
        Async generator that streams model events while preserving Thread semantics.
        Yields a normalized set of streaming events and finalized items.
        """
        self.verify_max_turns()

        if not self.client:
            self.client = AsyncOpenAI()

        if self.run_pipelines:
            await asyncio.gather(*[pipeline.preflight(self) for pipeline in self.run_pipelines])

        run_input = await self.get_run_input()

        run_context = self.create_run_context(run_input)

        if self.hooks:
            await asyncio.gather(*[hook.on_start(run_context) for hook in self.hooks])

        async with self.client.responses.stream(
            model=self.agent.model,
            instructions=self.agent.instructions,
            input=run_input,
            tools=self.tool_definitions(),
            temperature=self.agent.temperature,
            truncation="auto",
            text=self.get_output_format(),
            prompt_cache_key=self.prompt_cache_key,
            store=False,
            reasoning=Reasoning(effort=self.agent.reasoning_effort) if getattr(self.agent, 'reasoning_effort', None) else None,
        ) as s:
            async for event in s:
                et = getattr(event, "type", None)
                if et == "response.failed":
                    error = getattr(event, "error", "Streaming failed")
                    raise StreamingFailedException(str(error))

                yield event

            response = await s.get_final_response()

        self.input.extend(response.output)
        run_context.input.extend(response.output)

        if self.hooks:
            await asyncio.gather(*[hook.on_end(run_context, response) for hook in self.hooks])

        if self.run_pipelines:
            await asyncio.gather(*[pipeline.postflight(self, response) for pipeline in self.run_pipelines])

        if function_calls := self.collect_function_calls(response):
            async for output in self.execute_tool_calls(function_calls, run_context, next_turn_coro=self.stream):
                yield output
            
    async def run_to_completion(self):
        """
        Use this to get structured output.
        """
        final_output = None
        async for output in self.run():
            final_output = output
        
        if self.agent.output_type:
            final_output = await self.parse_structured_output(final_output)

        return final_output