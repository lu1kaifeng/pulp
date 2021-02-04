from typing import Optional

import uvicorn

from pulp.pipeline.QAPipeLine import QAPipeLine
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return pl("1702.00505.txt", "what is The goal of this exploration")


if __name__ == '__main__':
    pl = QAPipeLine()
    uvicorn.run(app, host="0.0.0.0", port=8000)
