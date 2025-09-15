from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fast_agents import Thread
    from openai.types.responses import Response

class RunPipeline(ABC):
    """
    - Allow you to modify thread during the runs.
    - Can you use for handoffs or changing reasoning effort / model mid-flight.
    - Changes to thread will persist for next runs as well.
    """

    async def preflight(self, thread: 'Thread'):
        """
        Input can be accessed or modified via thread.input.
        Model, reasoning effort, temperature, etc. can be modified as needed.
        """
        pass

    async def postflight(self, thread: 'Thread', response: 'Response'):
        """
        Response may contain function calls based on which next model can be selected (for handling handoffs).
        """
        pass