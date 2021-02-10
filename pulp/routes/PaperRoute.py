from fastapi import APIRouter
from fastapi import Body

from pulp.model.QAQuery import QAQuery
from pulp.pipeline.QAPipeLine import QAPipeLine
from . import META_STORE as meta_store
paper_route = APIRouter(prefix='/paper')


@paper_route.get("/{paper_id}")
def read_root(paper_id: str):
    return meta_store.get_all_documents('meta',{'id':[paper_id]},False)[0]
