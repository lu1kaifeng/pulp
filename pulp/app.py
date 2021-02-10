from typing import Optional

import uvicorn

from pulp.pipeline.QAPipeLine import QAPipeLine
from fastapi import FastAPI

from pulp.routes import QARoute, PaperRoute, FuzzyMatchRoute

app = FastAPI()
parent_app = FastAPI()
parent_app.mount('/api',app)
app.include_router(QARoute.qa_route)
app.include_router(PaperRoute.paper_route)
app.include_router(FuzzyMatchRoute.fuzzy_route)

if __name__ == '__main__':
    uvicorn.run(parent_app, host="0.0.0.0", port=8000)
