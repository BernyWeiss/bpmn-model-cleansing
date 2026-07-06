from pydantic import BaseModel, Field, AliasChoices, field_validator
from typing import List, Optional, Any


class Stensil(BaseModel):
    id: str = None


class Properties(BaseModel):
    name: Optional[str] = None
    documentation: Optional[str] = None
    text: Optional[str] = None
    callactivity: Optional[bool] = Field(default=None, validation_alias=AliasChoices('callactivity', 'callacitivity'))
    # tasktype: Optional[str] = None
    # stakeholder: Optional[str] = None
    # language: Optional[str] = None

    @field_validator("callactivity", mode="before")
    def convert_empty_str_to_none(cls, value: str) -> str|None:
        if value == '':
            return None
        return value


class ShapeReference(BaseModel):
    resourceId: str = None


class Shape(BaseModel):
    resourceId: str = None
    stencil: Optional[Stensil] = None
    properties: Properties = None
    childShapes: List['Shape'] = []
    outgoing: Optional[List[ShapeReference]] = []


def reduce_json_model(jsonstring: str) -> dict[str, Any]:
    raw_json_model = Shape.model_validate_json(jsonstring)
    small_json = raw_json_model.model_dump(exclude_unset=True)
    return small_json
