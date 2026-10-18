import pytest
from fastapi.testclient import TestClient

from searchlab.api import create_app


@pytest.fixture()
def client(engine):
    with TestClient(create_app(engine)) as client:
        yield client


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 6


def test_defaults_endpoint(client):
    defaults = client.get("/defaults").json()
    assert defaults["rrf_k"] == 60
    assert defaults["rerank"] is False


def test_compare_returns_three_panels_with_latency(client):
    body = client.post("/compare", json={"query": "pension eligibility", "top_k": 3}).json()
    assert [p["name"] for p in body["panels"]] == ["BM25", "Dense (HNSW)", "Hybrid"]
    for panel in body["panels"]:
        assert panel["latency_ms"] >= 0
        assert 1 <= len(panel["results"]) <= 3
        assert {"rank", "doc_id", "title", "snippet", "score"} <= panel["results"][0].keys()


def test_bm25_finds_the_pension_documents(client):
    body = client.post("/compare", json={"query": "pension", "top_k": 2}).json()
    bm25_ids = {r["doc_id"] for r in body["panels"][0]["results"]}
    assert bm25_ids == {"hr_pension", "hr_retire"}


def test_rerank_adds_a_fourth_panel(client):
    body = client.post(
        "/compare", json={"query": "pension plan", "rerank": True, "rerank_pool": 4, "top_k": 2}
    ).json()
    assert len(body["panels"]) == 4
    assert body["panels"][3]["name"].startswith("Hybrid + reranker")
    assert body["panels"][3]["results"][0]["doc_id"] == "hr_pension"


@pytest.mark.parametrize("method", ["rrf", "weighted_rrf", "score_fusion"])
def test_every_fusion_method_works(client, method):
    response = client.post("/compare", json={"query": "pension", "fusion_method": method})
    assert response.status_code == 200


def test_exact_dense_backend(client):
    body = client.post("/compare", json={"query": "pension", "dense_backend": "exact"}).json()
    assert body["panels"][1]["name"] == "Dense (exact)"


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "x", "top_k": 0},
        {"query": "x", "bm25_b": 2},
        {"query": "x", "fusion_method": "magic"},
        {"query": "x", "dense_backend": "faiss"},
    ],
)
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/compare", json=payload).status_code == 422


def test_blank_query_is_rejected(client):
    assert client.post("/compare", json={"query": "   "}).status_code == 422


def test_both_weights_zero_is_rejected(client):
    payload = {"query": "x", "fusion_method": "weighted_rrf", "bm25_weight": 0, "dense_weight": 0}
    assert client.post("/compare", json=payload).status_code == 422


class StubLLM:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user, temperature=0.0, max_tokens=300):
        return self.reply


@pytest.fixture()
def rag_client(engine):
    reply = "Employees become eligible for the company pension plan after one year of service [1]."
    with TestClient(create_app(engine, llm=StubLLM(reply))) as client:
        yield client


def test_answer_returns_cited_grounded_answer(rag_client):
    body = rag_client.post("/answer", json={"question": "pension eligibility", "top_k": 3}).json()
    assert body["answer"].endswith("[1].")
    assert body["cited"] == [1]
    assert body["grounded_ratio"] is not None
    assert body["sources"][0]["n"] == 1
    assert body["context_tokens"] <= body["token_budget"]


def test_answer_respects_the_token_budget(rag_client):
    body = rag_client.post("/answer", json={"question": "pension", "top_k": 5, "token_budget": 200}).json()
    assert body["context_tokens"] <= 200


def test_answer_with_reranking_enabled(rag_client):
    response = rag_client.post("/answer", json={"question": "pension plan", "rerank": True, "rerank_pool": 4})
    assert response.status_code == 200


def test_answer_without_an_llm_configured_is_a_clear_503(engine, monkeypatch, tmp_path):
    for var in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    with TestClient(create_app(engine)) as client:
        response = client.post("/answer", json={"question": "pension"})
    assert response.status_code == 503
    assert "LLM not configured" in response.json()["detail"]


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "x", "token_budget": 5}, {"question": "x", "top_k": 99}])
def test_answer_rejects_invalid_requests(rag_client, payload):
    assert rag_client.post("/answer", json=payload).status_code == 422
