import asyncio
import json
from typing import TYPE_CHECKING, Optional

from openai import AsyncOpenAI
from openai.types.responses import ResponseInputParam, ResponseTextConfigParam, ResponseFormatTextJSONSchemaConfigParam, \
    ResponseOutputItem
from pydantic import ValidationError

from fast_agents.exceptions import MaxTurnsReachedException, RefusalException, InvalidJSONResponseException, \
    InvalidPydanticSchemaResponseException
from fast_agents.helpers.llm_context_helper import gather_contexts
from fast_agents.helpers.schema_helper import format_parameters
from fast_agents.helpers.tokenisor import num_tokens_from_string
from fast_agents.run_context import RunContext
from fast_agents.tool_response import ToolResponse

if TYPE_CHECKING:
    from fast_agents.agent import Agent
    from fast_agents.llm_context import LlmContext
    from fast_agents.hook import Hook
    from pydantic import BaseModel

    
class Thread:
    def __init__(self,
                 agent: 'Agent',
                 input: ResponseInputParam = [],
                 max_turns: int = 20,
                 context: Optional['BaseModel'] = None,
                 llm_contexts: Optional[list['LlmContext']] = None,
                 hooks: Optional[list['Hook']] = None,
                 model: Optional[str] = None,
                 max_input_tokens: Optional[int] = None,
                 user_id: Optional[str] = None,
                 ):
        self.agent = agent
        self.max_turns = max_turns
        self.input = input
        self.context = context
        self.llm_contexts = llm_contexts
        self.hooks = hooks
        self.turn_count = 0 
        self.max_input_tokens = max_input_tokens
        self.output_type = self.agent.output_type
        self.model = model or self.agent.model
        self.user_id = user_id
        # Client will be initialized when needed in arun()

        if not self.model:
            raise ValueError("Please provide a model in the thread constructor or in the agent.")
        
    async def get_run_input(self) -> list[ResponseInputParam]:     
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

        return selected_inputs

    def get_output_format(self) -> ResponseTextConfigParam:
        if self.output_type:
            return ResponseTextConfigParam(
                format=ResponseFormatTextJSONSchemaConfigParam(
                    type="json_schema",
                    name=self.output_type.__name__,
                    description=self.output_type.__doc__,
                    schema=format_parameters(self.output_type),
                    strict=True
                )
            )

        return ResponseTextConfigParam(
            format={"type": "text"}
        )

    async def parse_structured_output(self, output: ResponseOutputItem) -> 'BaseModel':
        content = output.content[0]
        if content.type == "refusal":
            raise RefusalException(content.text)

        try:
            parsed_json = json.loads(content.text)
            return self.output_type(**parsed_json)
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
                    return f"[Error] Invalid JSON: {args}"

                # Create a new instance of the tool for each call
                response = await tool.__class__().arun(**parsed_args, run_context=run_context)
                return response

        return f"[Error] No tool found with name {name}"

    def verify_max_turns(self):
        if self.turn_count > self.max_turns:
            raise MaxTurnsReachedException()
        self.turn_count += 1

    async def run(self):
        self.verify_max_turns()
        run_input = await self.get_run_input()

        run_context = RunContext(
            agent=self.agent,
            turn=self.turn_count,
            max_turns=self.max_turns,
            model=self.model,
            input=run_input,
            context=self.context
        )

        if self.hooks:
            await asyncio.gather(*[hook.on_start(run_context) for hook in self.hooks])

        if not hasattr(self, 'client') or self.client is None:
            self.client = AsyncOpenAI()  # Initialize client when needed
        response = await self.client.responses.create(
            model=self.model,
            instructions=self.agent.instructions,
            input=run_input,
            tools=[tool.tool_definition for tool in self.agent.tools],
            temperature=self.agent.temperature,
            truncation="auto",
            text=self.get_output_format(),
            user=self.user_id
        )

        if self.hooks:
            await asyncio.gather(*[hook.on_end(run_context, response) for hook in self.hooks])

        # Append outputs to input and collect function calls for parallel execution
        function_calls = []
        for output in response.output:
            self.input.append(output)
            run_context.input.append(output)
            yield output

            if output.type == "function_call":
                function_calls.append((output.name, output.arguments, output.call_id))
                            
        # Execute all function calls in parallel if any exist
        if function_calls:
            tool_tasks = [self.call_tool(name, args, run_context) for name, args, _ in function_calls]
            tool_responses = await asyncio.gather(*tool_tasks)
            
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
            
            # Recursively run again if there were tool calls
            async for output in self.run():
                yield output
            
    async def run_to_completion(self):
        final_output = None
        async for output in self.run():
            final_output = output
        
        if self.output_type:
            final_output = await self.parse_structured_output(final_output)

        return final_output


            
