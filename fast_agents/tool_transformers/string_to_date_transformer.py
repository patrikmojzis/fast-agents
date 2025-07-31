import json
from datetime import datetime

from fast_agents.tool_transformer import ToolTransformer


class StringToDateTransformer(ToolTransformer):

    def __init__(self, path: list[str], date_format: list | str = None) -> None:
        if date_format is None:
            date_format = ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%SZ']

        self.path = path
        self.date_format = [date_format] if isinstance(date_format, str) else date_format

    async def transform(self, **kwargs) -> dict:
        data = kwargs
        for path in self.path:
            if path not in data:
                return kwargs

            data = data.get(path)

        if not data:
            return kwargs

        if isinstance(data, str):
            if self._is_convertible(data):
                data = self._convert(data)
            else:
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass

        if isinstance(data, dict):
            self._transform_dict(data)

        if isinstance(data, list):
            self._transform_list(data)

        built_path = data
        for key in reversed(self.path):
            built_path = {key: built_path}

        return {**kwargs, **built_path}

    def _transform_dict(self, data: dict):
        for key, value in list(data.items()):  # Use list() to allow modification during iteration
            if isinstance(value, dict) and "$date" in value:
                date_string = value["$date"]
                if self._is_convertible(date_string):
                    data[key] = self._convert(date_string)
            elif isinstance(value, str):
                if self._is_convertible(value):
                    data[key] = self._convert(value)
            elif isinstance(value, dict):
                self._transform_dict(value)
            elif isinstance(value, list):
                self._transform_list(value)

    def _transform_list(self, data: list):
        for i, item in enumerate(data):
            if isinstance(item, dict) and "$date" in item:
                date_string = item["$date"]
                if self._is_convertible(date_string):
                    data[i] = self._convert(date_string)
            elif isinstance(item, dict):
                self._transform_dict(item)
            elif isinstance(item, list):
                self._transform_list(item)

    def _is_convertible(self, date_string):
        for date_format in self.date_format:
            if self._is_convertible_by_format(date_string, date_format):
                return True
        return False

    def _is_convertible_by_format(self, date_string, date_format):
        try:
            datetime.strptime(date_string, date_format)
            return True
        except ValueError:
            return False

    def _convert(self, date_string):
        for date_format in self.date_format:
            if self._is_convertible_by_format(date_string, date_format):
                return datetime.strptime(date_string, date_format)

        raise ValueError(f"Date string {date_string} is not convertible to date format {self.date_format}")