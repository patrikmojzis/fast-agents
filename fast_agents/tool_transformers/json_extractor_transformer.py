import json
from typing import Any

from fast_agents.tool_transformer import ToolTransformer


class JsonExtractorTransformer(ToolTransformer):

    def __init__(self, path: list[str]) -> None:
        self.path = path

    async def transform(self, **kwargs) -> dict:
        """Finds id fields and converts them to ObjectId."""
        data = kwargs
        for path in self.path:
            if path not in data:
                return kwargs

            data = data.get(path)

        if not data:
            return kwargs

        data = self._extract(data)
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = self._extract(value)

            # Rebuild the nested dictionary structure with the parsed date
            built_path = data
            for key in reversed(self.path):
                built_path = {key: built_path}

            return {**kwargs, **built_path}
        else:
            # If data is not a dict (like a string), return original kwargs
            return kwargs


    def _extract(self, data: Any) -> dict:
        if isinstance(data, str):
            try:
                data = json.loads(data)
                return self._extract(data)
            except json.JSONDecodeError:
                pass

        return data