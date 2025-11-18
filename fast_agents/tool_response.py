import json
from typing import Union, Optional

from openai.types.responses import ResponseInputParam
from pydantic import field_validator, Field, BaseModel


class ToolResponse(BaseModel):

    output: Union[dict, str, int, list] = Field(..., description="Data to be sent to the agent from the tool as response.")
    is_error: bool = Field(False, description="Indicates if the response is an error.")
    additional_inputs: Optional[ResponseInputParam] = Field(None, description="Data to be appended after function call result to the input of the agent.")

    @field_validator('output')
    @classmethod
    def transform_str_to_dict(cls, v, values):
        return (v if isinstance(v, dict) else {"message": v}) if v else None

    @property
    def output_str(self) -> str:
        dump = json.dumps(self.model_dump(include={'output'}), default=str)
        if self.is_error:
            return f"[Error] {dump}"

        return dump
