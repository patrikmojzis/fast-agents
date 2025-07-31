from typing import Callable

from openai.types.responses import ResponseInputParam
from openai.types.responses.response_input_param import Message

from ai.helpers.function_helper import convert_to_dict


def filter_input(input: ResponseInputParam, filters: list[Callable[[ResponseInputParam], ResponseInputParam]]) -> ResponseInputParam:
    for filter in filters:
        input = filter(input)
    return input

def filter_files(input: ResponseInputParam) -> ResponseInputParam:
    filtered_input: ResponseInputParam = []
    
    for item in input:
        item = convert_to_dict(item)
        if item.get("type") == "message":
            # Create a copy of the message without input_file content
            filtered_content = []
            for content in item.get("content"):
                content = convert_to_dict(content)
                if content.get("type") != "input_file":
                    filtered_content.append(content)
            
            # Only add the message if it still has content after filtering
            if filtered_content:
                # Create a new message with filtered content
                filtered_message = Message(
                    id=item.get("id"),
                    role=item.get("role"),
                    content=filtered_content,
                    type=item.get("type"),
                    status=item.get("status")
                )
                filtered_input.append(filtered_message)
        else:
            # Keep non-message items as they are
            filtered_input.append(item)
    
    return filtered_input
    

def filter_function_calls(input: ResponseInputParam) -> ResponseInputParam:
    filtered_input: ResponseInputParam = []
    
    for item in input:
        item = convert_to_dict(item)
        if item.get("type") != "function_call" and item.get("type") != "function_call_output":
            filtered_input.append(item)
            
    return filtered_input
            