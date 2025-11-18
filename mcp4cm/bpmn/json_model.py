from pydantic import BaseModel
from typing import List, Optional


class Stensil(BaseModel):
    id: str = None

class Properties(BaseModel):
    name: Optional[str] = None
    #documentation: Optional[str] = None
    #tasktype: Optional[str] = None
    #stakeholder: Optional[str] = None
    #language: Optional[str] = None

class ShapeReference(BaseModel):
    resourceId: str = None

class Shape(BaseModel):
    resourceId: str = None
    stencil: Optional[Stensil] = None
    properties: Properties = None
    childShapes: List['Shape'] = []
    outgoing: Optional[List[ShapeReference]] = []


