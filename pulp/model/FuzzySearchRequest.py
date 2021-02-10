from typing import List

from pydantic import BaseModel, Field


class FuzzySearchRequest(BaseModel):
    text: str = Field(None, title='Target Text')
    query: List[str] = []
    max_l_dist: int = Field(32, title='Max L Dist')