from pydantic import BaseModel, Field


class QAQuery(BaseModel):
    question: str = Field(None, title='Question')
    id:str = Field(None,title='Paper Id')
