import pytest
import requests

from searchlab.llm import LLMClient, LLMError, load_dotenv
from searchlab.query_transform import STRATEGY_NAMES, transform


class FakeResponse:
    def __init__(self, status_code=200, content="hello", headers=None):
        self.status_code = status_code
        self._content = content
        self.headers = headers or {}
        self.text = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def make_client(tmp_path, **kwargs):
    return LLMClient("secret-key", "https://example.test/v1/", "m", cache_dir=tmp_path, min_interval=0, **kwargs)


def test_complete_sends_openai_style_request(monkeypatch, tmp_path):
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, json=json)
        return FakeResponse(content="  an answer \n")

    monkeypatch.setattr(requests, "post", fake_post)
    assert make_client(tmp_path).complete("sys", "user") == "an answer"
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer secret-key"
    assert seen["json"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]


def test_responses_are_cached_on_disk(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(1) or FakeResponse(content="x"))
    client = make_client(tmp_path)
    assert client.complete("s", "u") == "x"
    assert client.complete("s", "u") == "x"
    assert len(calls) == 1
    assert client.cache_hits == 1
    # a fresh client with the same cache dir also makes no call
    assert make_client(tmp_path).complete("s", "u") == "x"
    assert len(calls) == 1


def test_different_prompts_are_cached_separately(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(content="y"))
    client = make_client(tmp_path)
    client.complete("s", "u1")
    client.complete("s", "u2")
    assert client.calls_made == 2


def test_rate_limit_is_retried(monkeypatch, tmp_path):
    responses = iter([FakeResponse(429, headers={"retry-after": "0"}), FakeResponse(200, content="ok")])
    monkeypatch.setattr(requests, "post", lambda *a, **k: next(responses))
    monkeypatch.setattr("searchlab.llm.time.sleep", lambda s: None)
    assert make_client(tmp_path).complete("s", "u") == "ok"


def test_auth_failure_raises_without_leaking_the_key(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(401))
    with pytest.raises(LLMError) as error:
        make_client(tmp_path).complete("s", "u")
    assert "secret-key" not in str(error.value)


def test_gives_up_after_max_retries(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(503))
    monkeypatch.setattr("searchlab.llm.time.sleep", lambda s: None)
    with pytest.raises(LLMError):
        make_client(tmp_path, max_retries=3).complete("s", "u")


def test_from_env_reports_missing_settings(monkeypatch, tmp_path):
    for var in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        LLMClient.from_env()


def test_load_dotenv_does_not_override_existing(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text('# comment\nLLM_MODEL="from-file"\nOTHER=1\n', encoding="utf-8")
    monkeypatch.setenv("OTHER", "already")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    load_dotenv(env)
    import os

    assert os.environ["LLM_MODEL"] == "from-file"
    assert os.environ["OTHER"] == "already"


class StubLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def complete(self, system, user, temperature=0.0, max_tokens=300):
        self.prompts.append((system, user))
        return self.reply


def test_original_makes_no_llm_call():
    llm = StubLLM("unused")
    t = transform(llm, "statins and risk", "original")
    assert (t.bm25_text, t.dense_text) == ("statins and risk", "statins and risk")
    assert llm.prompts == []


def test_rewrite_applies_to_both_retrievers_and_cleans_output():
    t = transform(StubLLM('  "Statin therapy and cardiovascular risk"\n'), "statins?", "rewrite")
    assert t.bm25_text == t.dense_text == "Statin therapy and cardiovascular risk"


def test_hyde_changes_only_the_dense_side():
    t = transform(StubLLM("Statins reduce LDL. They lower risk."), "do statins help", "hyde")
    assert t.bm25_text == "do statins help"
    assert t.dense_text == "Statins reduce LDL. They lower risk."


def test_expansion_appends_terms_to_the_original_query():
    t = transform(StubLLM("HMG-CoA, atorvastatin"), "statins", "expansion")
    assert t.bm25_text == t.dense_text == "statins HMG-CoA, atorvastatin"


def test_empty_llm_output_falls_back_to_the_original_query():
    assert transform(StubLLM("   "), "my query", "rewrite").bm25_text == "my query"


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        transform(StubLLM("x"), "q", "magic")


def test_all_listed_strategies_are_supported():
    for name in STRATEGY_NAMES:
        assert transform(StubLLM("something"), "q", name).strategy == name
