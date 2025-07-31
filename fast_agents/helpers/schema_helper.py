from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from pydantic import BaseModel


def remove_titles(d):
    if isinstance(d, dict):
        d.pop('title', None)
        for value in d.values():
            remove_titles(value)
    elif isinstance(d, list):
        for item in d:
            remove_titles(item)


def format_parameters(s: 'Type[BaseModel]'):
    schema = s.model_json_schema()
    remove_titles(schema)
    
    # Add additionalProperties: false to prevent extra properties
    if isinstance(schema, dict):
        schema['additionalProperties'] = False
    
    return schema

