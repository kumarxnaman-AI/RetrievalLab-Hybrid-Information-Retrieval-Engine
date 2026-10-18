"""FastAPI backend for the retrieval lab.  Run:  uvicorn searchlab.api:app --port 8765"""

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from searchlab.engine import SearchEngine, SearchParams
from searchlab.logging_setup import setup_logging

logger = logging.getLogger(__name__)


class CompareRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=20)
    bm25_k1: float = Field(1.5, ge=0, le=5)
    bm25_b: float = Field(0.75, ge=0, le=1)
    ef_search: int = Field(100, ge=10, le=1000)
    dense_backend: Literal["hnsw", "exact"] = "hnsw"
    fusion_method: Literal["rrf", "weighted_rrf", "score_fusion"] = "rrf"
    rrf_k: int = Field(60, ge=1, le=500)
    bm25_weight: float = Field(0.5, ge=0, le=1)
    dense_weight: float = Field(0.5, ge=0, le=1)
    alpha: float = Field(0.5, ge=0, le=1)
    rerank: bool = False
    rerank_pool: int = Field(20, ge=1, le=100)

    def to_params(self) -> SearchParams:
        data = self.model_dump(exclude={"query"})
        return SearchParams(**data)


def create_app(engine: SearchEngine | None = None) -> FastAPI:
    """Build the app. Pass an engine (tests) or let it load the dataset at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging("INFO")
        if engine is not None:
            app.state.engine = engine
        else:
            dataset = Path(os.environ.get("SEARCHLAB_DATASET", "datasets/scifact"))
            logger.info("loading engine from %s", dataset)
            app.state.engine = SearchEngine.from_dataset(dataset)
        app.state.engine.warm_up()
        logger.info("engine ready")
        yield

    app = FastAPI(title="RetrievalLab", version="0.1.0", lifespan=lifespan)

    def get_engine(request: Request) -> SearchEngine:
        return request.app.state.engine

    @app.get("/health")
    def health(request: Request):
        info = get_engine(request).info()
        return {"status": "ok", **info}

    @app.get("/defaults")
    def defaults():
        return asdict(SearchParams())

    @app.post("/compare")
    def compare(body: CompareRequest, request: Request):
        """Run BM25, dense and hybrid (and optionally hybrid + reranker) on one query."""
        if not body.query.strip():
            raise HTTPException(status_code=422, detail="query must not be blank")
        if body.fusion_method == "weighted_rrf" and body.bm25_weight + body.dense_weight == 0:
            raise HTTPException(status_code=422, detail="weights must not both be zero")
        panels = get_engine(request).compare(body.query, body.to_params())
        return {"query": body.query, "panels": [asdict(p) for p in panels]}

    return app


app = create_app()
