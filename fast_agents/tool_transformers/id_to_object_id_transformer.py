import json

from bson import ObjectId

from fast_agents.tool_transformer import ToolTransformer


class IdToObjectIdTransformer(ToolTransformer):

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

        if isinstance(data, str):
            if ObjectId.is_valid(data):
                data = ObjectId(data)
            else:
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass

        if isinstance(data, dict):
            self._transform_dict(data)

        # loop through list
        if isinstance(data, list):
            self._transform_list(data)

        # Rebuild the nested dictionary structure with the parsed date
        built_path = data
        for key in reversed(self.path):
            built_path = {key: built_path}

        return {**kwargs, **built_path}

    def _transform_dict(self, data: dict):
        for key, value in data.items():
            if isinstance(value, str):
                if ObjectId.is_valid(value):
                    if key != "product_id":
                        data[key] = ObjectId(value)
            elif isinstance(value, dict):
                self._transform_dict(value)
            elif isinstance(value, list):
                self._transform_list(value)

    def _transform_list(self, data: list):
        for i in range(len(data)):
            if isinstance(data[i], dict):
                self._transform_dict(data[i])
            if isinstance(data[i], list):
                self._transform_list(data[i])
