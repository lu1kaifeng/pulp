from fastapi import APIRouter
from fastapi import Body

from pulp.model.QAQuery import QAQuery
from pulp.pipeline.QAPipeLine import QAPipeLine
from . import QA_PIPELINE as pl
qa_route = APIRouter(prefix='/qa')



@qa_route.post("/question")
def read_root(query: QAQuery = Body(..., embed=True)):
    return pl(query.id+'.txt', query.question)
