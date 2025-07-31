from typing import Optional

from fast_agents.tool_transformer import ToolTransformer


class SortTransformer(ToolTransformer):
    async def transform(self, sort: Optional[list[dict[str, int]]], **kwargs) -> dict:
        """Transforms sort from list of dicts to list of tuples."""
        return {**kwargs, "sort": [(list(s.keys())[0], list(s.values())[0]) for s in sort] if sort else None}
