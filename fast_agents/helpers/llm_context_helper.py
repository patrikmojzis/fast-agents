import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fast_agents.llm_context import LlmContext

async def gather_contexts(contexts: list['LlmContext']) -> str:
    results = await asyncio.gather(*(context.dumps() for context in contexts))
    return '\n\n'.join(results)