from pathlib import Path

import requests
from streamlit.testing.v1 import AppTest

DASHBOARD = str(Path(__file__).resolve().parent.parent / "src" / "searchlab" / "dashboard.py")

FAKE_RESPONSE = {
    "query": "pension",
    "panels": [
        {
            "name": name,
            "score_label": label,
            "latency_ms": 3.2,
            "results": [
                {"rank": 1, "doc_id": "d1", "title": "Pension plan", "snippet": "Eligible after a year.", "score": 1.5}
            ],
        }
        for name, label in [("BM25", "score"), ("Dense (HNSW)", "similarity"), ("Hybrid", "RRF score")]
    ],
}


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def run_app():
    at = AppTest.from_file(DASHBOARD, default_timeout=30)
    at.run()
    return at


def submit(at):
    next(b for b in at.button if b.label == "Search").click().run()
    return at


def test_page_loads_without_errors():
    at = run_app()
    assert not at.exception
    assert at.title[0].value == "RetrievalLab"


def test_search_renders_one_panel_per_retriever(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(json)
        return FakeResponse(200, FAKE_RESPONSE)

    monkeypatch.setattr(requests, "post", fake_post)
    at = submit(run_app())

    assert not at.exception
    assert [s.value for s in at.main.subheader if s.value in ("BM25", "Dense (HNSW)", "Hybrid")] == [
        "BM25",
        "Dense (HNSW)",
        "Hybrid",
    ]
    assert sent["top_k"] == 5 and sent["rerank"] is False and sent["fusion_method"] == "rrf"


def test_settings_are_sent_to_the_api(monkeypatch):
    sent = {}
    monkeypatch.setattr(requests, "post", lambda url, json, timeout: sent.update(json) or FakeResponse(200, FAKE_RESPONSE))

    at = run_app()
    at.sidebar.slider[0].set_value(3)  # Top K
    at.sidebar.toggle[0].set_value(True)  # reranker on
    submit(at)

    assert sent["top_k"] == 3
    assert sent["rerank"] is True


def test_api_down_shows_helpful_error(monkeypatch):
    def refuse(url, json, timeout):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(requests, "post", refuse)
    at = submit(run_app())
    assert any("Cannot reach the API" in e.value for e in at.error)


def test_api_validation_error_is_shown(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda url, json, timeout: FakeResponse(422, {"detail": "bad"}))
    at = submit(run_app())
    assert any("422" in e.value for e in at.error)


ANSWER = {
    "question": "q",
    "answer": "Statins lower risk [1].",
    "sources": [{"n": 1, "doc_id": "d1", "title": "Statins", "passage": "Statins lower risk.", "score": 0.9}],
    "context_tokens": 120,
    "token_budget": 1500,
    "cited": [1],
    "invalid_citations": [],
    "unsupported_sentences": [],
    "grounded_ratio": 1.0,
    "abstained": False,
}


def ask(at):
    next(b for b in at.button if b.label == "Answer").click().run()
    return at


def test_ask_tab_shows_answer_grounding_and_sources(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent["url"] = url
        sent.update(json)
        return FakeResponse(200, ANSWER)

    monkeypatch.setattr(requests, "post", fake_post)
    at = ask(run_app())

    assert not at.exception
    assert sent["url"].endswith("/answer")
    assert sent["token_budget"] == 1500 and "question" in sent
    assert any("Statins lower risk [1]." in m.value for m in at.markdown)
    labels = [m.label for m in at.metric]
    assert labels == ["Grounded sentences", "Context used", "Sources"]
    assert at.metric[0].value == "100%"


def test_ask_tab_warns_about_unsupported_sentences(monkeypatch):
    bad = dict(ANSWER, unsupported_sentences=["Made up claim."], invalid_citations=[7], grounded_ratio=0.5)
    monkeypatch.setattr(requests, "post", lambda url, json, timeout: FakeResponse(200, bad))
    at = ask(run_app())
    assert any("Made up claim." in w.value for w in at.warning)
    assert any("[7]" in e.value for e in at.error)


def test_ask_tab_shows_llm_not_configured_error(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda url, json, timeout: FakeResponse(503, {"detail": "LLM not configured"})
    )
    at = ask(run_app())
    assert any("503" in e.value for e in at.error)
