from typing import Any

from openai.types.responses import ResponseInputItemParam


def response_to_dict(item: Any) -> dict:
    # Handle different input types more robustly
    if isinstance(item, dict):
        return item
    elif hasattr(item, 'model_dump'):
        return item.model_dump()
    elif hasattr(item, '__dict__'):
        return vars(item)
    elif hasattr(item, '__iter__') and not isinstance(item, str):
        return dict(item)
    else:
        raise ValueError(f"Cannot convert item to dict: {item}")


def string_to_user_message(message: str) -> ResponseInputItemParam:
    return {
        "role": "user",
        "type": "message",
        "content": [
            {
                "type": "input_text",
                "text": message
            }
        ]
    }