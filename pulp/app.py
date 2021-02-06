from typing import Optional

import uvicorn

from pulp.pipeline.QAPipeLine import QAPipeLine
from fastapi import FastAPI

from pulp.routes import QARoute

app = FastAPI()

app.include_router(QARoute.qa_route)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000,root_path='/api')
