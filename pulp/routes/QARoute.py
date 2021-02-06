from fastapi import APIRouter
from fastapi import Body

from pulp.model.QAQuery import QAQuery
from pulp.pipeline.QAPipeLine import QAPipeLine

qa_route = APIRouter(prefix='/qa')
pl = QAPipeLine()


@qa_route.post("/question")
def read_root(query: QAQuery = Body(..., embed=True)):
    return pl(query.id, query.question)
