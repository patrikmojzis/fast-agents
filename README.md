# Fast Agents

A clean, standalone AI framework package for building AI agents with OpenAI's API.

**Author:** Patrik Mojzis  
**Website:** https://fast-agents.patrikmojzis.com  
**Repository:** https://github.com/patrikmojzis/fast-agents

## Features

- Simple and elegant agent framework
- Tool-based extensibility
- Built-in validation and transformation
- Async/await support
- Type-safe with Pydantic models

## Installation

### Local Development Installation

```bash
pip install -e .
```

### From PyPI (when published)

```bash
pip install fast-agents
```

## Quick Start

```python
from fast_agents import Agent, Tool, ToolResponse

# Create a custom tool
class MyTool(Tool):
    name = "my_tool"
    description = "A sample tool"
    
    async def handle(self, **kwargs) -> ToolResponse:
        return ToolResponse(output="Hello from my tool!")

# Create an agent
agent = Agent(
    name="my_agent",
    instructions="You are a helpful assistant.",
    tools=[MyTool()]
)

# Use the agent in your application
```

## Dependencies

- `pydantic>=2.0.0` - Data validation and settings management
- `openai>=1.0.0` - OpenAI API client
- `tiktoken>=0.4.0` - Token counting for OpenAI models

## License

MIT License