from fastapi import APIRouter
from fastapi import Body
from fuzzysearch import find_near_matches
from . import META_STORE as meta_store
from pulp.model.FuzzySearchRequest import FuzzySearchRequest

fuzzy_route = APIRouter(prefix='/fuzzy')


@fuzzy_route.post("/search")
def read_root(query: FuzzySearchRequest):
    l = []
    for q in query.query:
        l.append(find_near_matches(q,query.text,max_l_dist=query.max_l_dist))
    l = list(map(lambda x:{'start':x[0].start,'end':x[0].end,'matched':x[0].matched},l))
    return l
