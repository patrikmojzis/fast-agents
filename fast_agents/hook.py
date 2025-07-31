from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from fast_agents.run_context import RunContext
    from openai.types.responses import Response


class Hook(BaseModel):
    async def on_start(self, run_context: 'RunContext'):
        pass

    async def on_end(self, run_context: 'RunContext', output: 'Response'):
        pass
