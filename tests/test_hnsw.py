import numpy as np
import pytest

from searchlab.index.hnsw import HNSWIndex


def random_data(n=600, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)


def build_index(vectors, **kwargs):
    params = dict(M=8, ef_construction=64, ef_search=50, seed=1)
    params.update(kwargs)
    index = HNSWIndex(dim=vectors.shape[1], max_elements=len(vectors), **params)
    index.build(vectors, [f"v{i}" for i in range(len(vectors))])
    return index


def exact_top_k(vectors, query, k):
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = unit @ (query / np.linalg.norm(query))
    return [f"v{i}" for i in np.argsort(-sims)[:k]]


def test_recall_against_brute_force_is_high():
    vectors = random_data()
    index = build_index(vectors)
    queries = random_data(n=40, seed=99)

    hits = 0
    for q in queries:
        truth = set(exact_top_k(vectors, q, 10))
        found = {n.document_id for n in index.search(q, k=10, ef_search=100)}
        hits += len(truth & found)

    assert hits / (40 * 10) >= 0.95


def test_larger_ef_search_does_not_reduce_recall():
    vectors = random_data(n=800, dim=24)
    index = build_index(vectors, M=6, ef_construction=40)
    queries = random_data(n=40, dim=24, seed=5)

    def recall(ef):
        total = 0
        for q in queries:
            truth = set(exact_top_k(vectors, q, 10))
            total += len(truth & {n.document_id for n in index.search(q, 10, ef_search=ef)})
        return total / (len(queries) * 10)

    assert recall(200) >= recall(10)


def test_exact_match_is_found_first_with_zero_distance():
    vectors = random_data()
    index = build_index(vectors)
    best = index.search(vectors[123], k=1)[0]
    assert best.document_id == "v123"
    assert best.distance == pytest.approx(0.0, abs=1e-5)


def test_results_are_sorted_nearest_first():
    vectors = random_data()
    index = build_index(vectors)
    distances = [n.distance for n in index.search(random_data(1, seed=7)[0], k=10)]
    assert distances == sorted(distances)


def test_degree_limits_are_respected():
    vectors = random_data()
    index = build_index(vectors)
    for node in index.nodes:
        assert len(node.neighbours) == node.level + 1
        assert len(node.neighbours[0]) <= index.max_m0
        for layer in range(1, node.level + 1):
            assert len(node.neighbours[layer]) <= index.M


def test_graph_has_no_self_loops_or_duplicate_links():
    index = build_index(random_data())
    for node_id, node in enumerate(index.nodes):
        for links in node.neighbours:
            assert node_id not in links
            assert len(links) == len(set(links))


def test_layers_shrink_geometrically():
    index = build_index(random_data(n=1500), M=8)
    per_layer = index.stats()["nodes_per_layer"]
    assert per_layer[0] == 1500
    assert all(a > b for a, b in zip(per_layer, per_layer[1:]))


def test_entry_point_is_on_the_top_layer():
    index = build_index(random_data())
    assert index.nodes[index.entry_point].level == index.max_level


def test_same_seed_gives_same_graph():
    vectors = random_data(n=200)
    a = build_index(vectors, seed=3)
    b = build_index(vectors, seed=3)
    assert [n.neighbours for n in a.nodes] == [n.neighbours for n in b.nodes]


def test_simple_selection_mode_also_works():
    vectors = random_data(n=300)
    index = build_index(vectors, use_heuristic=False)
    assert index.search(vectors[5], k=1)[0].document_id == "v5"


def test_l2_metric():
    vectors = random_data(n=300)
    index = build_index(vectors, metric="l2")
    best = index.search(vectors[10], k=1)[0]
    assert best.document_id == "v10"
    assert best.distance == pytest.approx(0.0, abs=1e-5)


def test_search_empty_index_returns_nothing():
    assert HNSWIndex(dim=4, max_elements=10).search(np.ones(4), k=3) == []


def test_k_larger_than_index_returns_everything():
    vectors = random_data(n=5, dim=4)
    index = build_index(vectors, M=2, ef_construction=4)
    assert len(index.search(vectors[0], k=20)) == 5


def test_index_full_raises():
    index = HNSWIndex(dim=2, max_elements=1)
    index.add(np.array([1.0, 0.0]), "a")
    with pytest.raises(ValueError):
        index.add(np.array([0.0, 1.0]), "b")


def test_wrong_dimension_raises():
    index = HNSWIndex(dim=3, max_elements=5)
    with pytest.raises(ValueError):
        index.add(np.ones(4), "a")


@pytest.mark.parametrize(
    "kwargs",
    [dict(M=1), dict(metric="manhattan"), dict(ef_construction=2, M=8), dict(ef_search=0)],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        HNSWIndex(dim=4, max_elements=10, **kwargs)


def test_save_and_load_roundtrip(tmp_path):
    vectors = random_data(n=200)
    index = build_index(vectors)
    index.save(tmp_path / "index.pkl")
    loaded = HNSWIndex.load(tmp_path / "index.pkl")

    query = random_data(1, seed=8)[0]
    assert index.search(query, k=5) == loaded.search(query, k=5)
    assert loaded.stats() == index.stats()
