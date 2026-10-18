"""Streamlit retrieval lab.  Run:  streamlit run src/searchlab/dashboard.py

Needs the API running (uvicorn searchlab.api:app --port 8765).
"""

import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.environ.get("SEARCHLAB_API_URL", "http://127.0.0.1:8765")
RESULTS_DIR = Path(os.environ.get("SEARCHLAB_RESULTS", "results"))
EXAMPLES = [
    "do statins reduce risk of cardiovascular disease",
    "vitamin D deficiency and mortality",
    "BRCA1 mutations and breast cancer risk",
]

st.set_page_config(page_title="RetrievalLab", layout="wide")
st.title("RetrievalLab")
st.caption("BM25 vs dense (our HNSW) vs hybrid vs hybrid + cross-encoder reranker, on the same query.")


def sidebar_controls() -> dict:
    st.sidebar.header("Settings")
    st.sidebar.subheader("Results")
    top_k = st.sidebar.slider("Top K", 1, 10, 5)

    st.sidebar.subheader("BM25")
    k1 = st.sidebar.slider("k1 (term-frequency saturation)", 0.0, 3.0, 1.5, 0.1)
    b = st.sidebar.slider("b (length normalization)", 0.0, 1.0, 0.75, 0.05)

    st.sidebar.subheader("Dense")
    backend = st.sidebar.radio("Vector search", ["hnsw", "exact"], horizontal=True)
    ef = st.sidebar.slider("HNSW efSearch", 10, 500, 100, 10, disabled=backend == "exact")

    st.sidebar.subheader("Hybrid fusion")
    method = st.sidebar.selectbox(
        "Method", ["rrf", "weighted_rrf", "score_fusion"], format_func=lambda m: {
            "rrf": "Reciprocal Rank Fusion",
            "weighted_rrf": "Weighted RRF",
            "score_fusion": "Normalized score fusion",
        }[m],
    )
    rrf_k = st.sidebar.slider("RRF k", 1, 200, 60, disabled=method == "score_fusion")
    bm25_w = st.sidebar.slider("BM25 weight", 0.0, 1.0, 0.5, 0.05, disabled=method != "weighted_rrf")
    dense_w = st.sidebar.slider("Dense weight", 0.0, 1.0, 0.5, 0.05, disabled=method != "weighted_rrf")
    alpha = st.sidebar.slider("alpha (dense share)", 0.0, 1.0, 0.5, 0.05, disabled=method != "score_fusion")

    st.sidebar.subheader("Reranking")
    rerank = st.sidebar.toggle("Cross-encoder reranker", value=False)
    pool = st.sidebar.slider("Rerank top N candidates", 5, 100, 20, 5, disabled=not rerank)

    return {
        "top_k": top_k,
        "bm25_k1": k1,
        "bm25_b": b,
        "dense_backend": backend,
        "ef_search": ef,
        "fusion_method": method,
        "rrf_k": rrf_k,
        "bm25_weight": bm25_w,
        "dense_weight": dense_w,
        "alpha": alpha,
        "rerank": rerank,
        "rerank_pool": pool,
    }


def post_api(path: str, payload: dict):
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=300)
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot reach the API at {API_URL}. Start it with: "
            "`uvicorn searchlab.api:app --port 8765`"
        )
        return None
    if response.status_code != 200:
        st.error(f"API error {response.status_code}: {response.text}")
        return None
    return response.json()


def call_api(query: str, settings: dict):
    return post_api("/compare", {"query": query, **settings})


def render_panel(panel: dict) -> None:
    st.subheader(panel["name"])
    st.caption(f"{panel['latency_ms']:.0f} ms  ·  ranked by {panel['score_label']}")
    for r in panel["results"]:
        st.markdown(f"**{r['rank']}. {r['title']}**")
        st.caption(f"doc {r['doc_id']}  ·  {panel['score_label']}: {r['score']:.3f}")
        st.write(r["snippet"])


def render_search_tab(settings: dict) -> None:
    if "query" not in st.session_state:
        st.session_state["query"] = EXAMPLES[0]
    st.write("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, key=f"ex-{example}"):
            st.session_state["query"] = example

    with st.form("search"):
        query = st.text_input("Query", key="query")
        submitted = st.form_submit_button("Search", type="primary")

    if not submitted:
        st.info("Enter a query and press Search.")
        return
    if not query.strip():
        st.warning("Please enter a query.")
        return

    with st.spinner("Searching..."):
        data = call_api(query, settings)
    if data is None:
        return

    panels = data["panels"]
    for col, panel in zip(st.columns(len(panels)), panels):
        with col:
            render_panel(panel)

    top = {p["name"]: {r["doc_id"] for r in p["results"]} for p in panels}
    names = list(top)
    shared = top[names[0]] & top[names[1]]
    st.caption(f"{len(shared)} of the top {settings['top_k']} documents are shared by {names[0]} and {names[1]}.")


def render_answer_tab(settings: dict) -> None:
    st.write(
        "Ask a question. The system retrieves with the hybrid settings in the sidebar, gives the LLM "
        "numbered sources, and checks that every answer sentence is cited and supported."
    )
    with st.form("ask"):
        question = st.text_input("Question", value=EXAMPLES[0], key="question")
        budget = st.slider("Context token budget", 200, 4000, 1500, 100)
        submitted = st.form_submit_button("Answer", type="primary")
    if not submitted:
        return
    if not question.strip():
        st.warning("Please enter a question.")
        return

    with st.spinner("Retrieving and generating..."):
        data = post_api("/answer", {"question": question, **settings, "token_budget": budget})
    if data is None:
        return

    if data["abstained"]:
        st.warning("The model said it cannot answer from the retrieved sources.")
    st.markdown("### Answer")
    st.write(data["answer"])

    ratio = data["grounded_ratio"]
    cols = st.columns(3)
    cols[0].metric("Grounded sentences", "n/a" if ratio is None else f"{ratio:.0%}")
    cols[1].metric("Context used", f"{data['context_tokens']} / {data['token_budget']} tokens")
    cols[2].metric("Sources", len(data["sources"]))
    if data["invalid_citations"]:
        st.error(f"The answer cites sources that do not exist: {data['invalid_citations']}")
    for sentence in data["unsupported_sentences"]:
        st.warning(f"Not supported by its citations: {sentence}")

    st.markdown("### Sources")
    for source in data["sources"]:
        cited = source["n"] in data["cited"]
        with st.expander(f"[{source['n']}] {source['title']}" + ("  (cited)" if cited else ""), expanded=cited):
            st.caption(f"doc {source['doc_id']}  ·  retrieval score {source['score']:.3f}")
            st.write(source["passage"])


def render_results_tab() -> None:
    st.write("Results from the benchmark scripts in `results/` (all on BEIR SciFact).")
    sections = [
        ("Phase 1: BM25 vs exact dense search", "benchmark.md"),
        ("Phase 2: HNSW vs exact KNN", "hnsw_benchmark.md"),
        ("Phase 3: hybrid retrieval and fusion", "hybrid_benchmark.md"),
        ("Phase 4: cross-encoder reranking", "rerank_benchmark.md"),
        ("Phase 6: query rewriting, HyDE, expansion", "query_transform_benchmark.md"),
        ("Phase 7: chunking strategies", "chunking_benchmark.md"),
        ("Phase 9: multi-dataset benchmark and failure analysis", "multi_dataset_benchmark.md"),
        ("Phase 10: when does query rewriting harm retrieval?", "rewrite_analysis.md"),
    ]
    for title, filename in sections:
        with st.expander(title, expanded=filename == "hnsw_benchmark.md"):
            path = RESULTS_DIR / filename
            if path.exists():
                st.markdown(path.read_text(encoding="utf-8"))
                if filename == "hnsw_benchmark.md":
                    image = RESULTS_DIR / "hnsw_recall_vs_latency.png"
                    if image.exists():
                        st.image(str(image))
            else:
                st.info(f"{filename} not found. Run the matching benchmark first.")


search_tab, ask_tab, results_tab = st.tabs(["Search lab", "Ask (RAG)", "Experiment results"])
settings = sidebar_controls()
with search_tab:
    render_search_tab(settings)
with ask_tab:
    render_answer_tab(settings)
with results_tab:
    render_results_tab()
