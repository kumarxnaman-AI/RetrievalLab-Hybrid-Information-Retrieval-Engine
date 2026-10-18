"""HNSW (Hierarchical Navigable Small World) approximate nearest-neighbour index.

Written from scratch after Malkov & Yashunin (2016), "Efficient and robust approximate
nearest neighbor search using Hierarchical Navigable Small World graphs".

Idea in one paragraph: every vector is a node in a graph. Nodes are linked to nearby
nodes. A node also lives on several "layers": layer 0 holds every node, layer 1 holds
roughly 1/M of them, layer 2 roughly 1/M^2, and so on. A search enters at the top layer,
greedily walks toward the query (long hops), drops a layer, and repeats until layer 0,
where it explores more carefully (a beam of width `ef`) to collect the final neighbours.
"""

import heapq
import logging
import math
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

METRICS = ("cosine", "l2")


@dataclass
class Node:
    vector: np.ndarray  # a view into the index's vector matrix (not a copy)
    document_id: str
    level: int  # highest layer this node appears on
    neighbours: list[list[int]]  # neighbours[layer] -> ids of linked nodes


class Neighbour(NamedTuple):
    document_id: str
    distance: float  # cosine: 1 - similarity | l2: squared Euclidean distance


class HNSWIndex:
    def __init__(
        self,
        dim: int,
        max_elements: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        metric: str = "cosine",
        seed: int = 42,
        use_heuristic: bool = True,
    ) -> None:
        """
        M               links kept per node (layer 0 keeps up to 2*M). More = better recall,
                        more memory, slower build.
        ef_construction beam width while inserting. More = better graph, slower build.
        ef_search       default beam width while querying. More = better recall, slower.
        metric          "cosine" or "l2".
        use_heuristic   use the paper's diversity-aware neighbour selection (Algorithm 4)
                        instead of simply keeping the closest M.
        """
        if dim <= 0 or max_elements <= 0:
            raise ValueError("dim and max_elements must be positive")
        if M < 2:
            raise ValueError("M must be at least 2")
        if ef_construction < M:
            raise ValueError("ef_construction must be >= M")
        if ef_search <= 0:
            raise ValueError("ef_search must be positive")
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")

        self.dim = dim
        self.max_elements = max_elements
        self.M = M
        self.max_m0 = 2 * M  # layer 0 is denser, so it is allowed more links
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.metric = metric
        self.use_heuristic = use_heuristic
        self.seed = seed

        self._level_mult = 1.0 / math.log(M)
        self._rng = random.Random(seed)
        self._vectors = np.zeros((max_elements, dim), dtype=np.float32)
        self.nodes: list[Node] = []
        self.entry_point: int | None = None
        self.max_level = -1

    def __len__(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------------ distances

    def _prepare(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.dim,):
            raise ValueError(f"expected vector of shape ({self.dim},), got {vector.shape}")
        if self.metric == "cosine":
            # With unit vectors, cosine distance is simply 1 - dot product.
            vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        return vector

    def _dist_many(self, query: np.ndarray, ids: list[int]) -> np.ndarray:
        """Distances from `query` to several nodes in one vectorised call."""
        vectors = self._vectors[ids]
        if self.metric == "cosine":
            return 1.0 - vectors @ query
        diff = vectors - query
        return np.einsum("ij,ij->i", diff, diff)

    # ------------------------------------------------------------------ building

    def _random_level(self) -> int:
        """Exponentially decaying layer assignment: P(level >= l) = M**-l."""
        return int(-math.log(1.0 - self._rng.random()) * self._level_mult)

    def _search_layer(
        self, query: np.ndarray, entry_points: list[tuple[float, int]], ef: int, layer: int
    ) -> list[tuple[float, int]]:
        """Best-first beam search inside one layer.

        Returns up to `ef` (distance, node_id) pairs sorted nearest first.
        """
        visited = {node_id for _, node_id in entry_points}
        candidates = list(entry_points)  # min-heap: closest unexplored node first
        heapq.heapify(candidates)
        results = [(-dist, node_id) for dist, node_id in entry_points]  # max-heap
        heapq.heapify(results)

        while candidates:
            dist_c, current = heapq.heappop(candidates)
            # Everything left to explore is farther than our worst kept result: done.
            if len(results) >= ef and dist_c > -results[0][0]:
                break

            fresh = [n for n in self.nodes[current].neighbours[layer] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)

            worst = -results[0][0]
            for node_id, dist in zip(fresh, self._dist_many(query, fresh).tolist()):
                if len(results) < ef or dist < worst:
                    heapq.heappush(candidates, (dist, node_id))
                    heapq.heappush(results, (-dist, node_id))
                    if len(results) > ef:
                        heapq.heappop(results)
                    worst = -results[0][0]

        return sorted((-neg_dist, node_id) for neg_dist, node_id in results)

    def _select_neighbours(self, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """Choose up to m links from candidates (sorted nearest first).

        Simple mode keeps the m closest. Heuristic mode (Algorithm 4) keeps a candidate
        only if it is closer to the base point than to every neighbour already chosen.
        This spreads links in different directions, which keeps clusters connected.
        """
        if len(candidates) <= m:
            return [node_id for _, node_id in candidates]
        if not self.use_heuristic:
            return [node_id for _, node_id in candidates[:m]]

        selected: list[int] = []
        pruned: list[int] = []
        for dist_to_base, node_id in candidates:
            if len(selected) >= m:
                break
            if selected:
                to_selected = self._dist_many(self._vectors[node_id], selected)
                if float(to_selected.min()) < dist_to_base:
                    pruned.append(node_id)  # a chosen neighbour already covers it
                    continue
            selected.append(node_id)

        # keepPrunedConnections: top up with the best pruned candidates.
        for node_id in pruned:
            if len(selected) >= m:
                break
            selected.append(node_id)
        return selected

    def add(self, vector: np.ndarray, document_id: str) -> int:
        """Insert one vector. Returns its internal node id."""
        if len(self.nodes) >= self.max_elements:
            raise ValueError(f"index is full (max_elements={self.max_elements})")

        vector = self._prepare(vector)
        node_id = len(self.nodes)
        self._vectors[node_id] = vector
        level = self._random_level()
        node = Node(self._vectors[node_id], document_id, level, [[] for _ in range(level + 1)])
        self.nodes.append(node)

        if self.entry_point is None:
            self.entry_point, self.max_level = node_id, level
            return node_id

        entry = self.entry_point
        nearest = [(float(self._dist_many(vector, [entry])[0]), entry)]

        # Phase 1: layers above the new node's level. Just walk greedily downwards.
        for layer in range(self.max_level, level, -1):
            nearest = self._search_layer(vector, nearest, 1, layer)

        # Phase 2: from the node's level down to 0, find neighbours and link both ways.
        for layer in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(vector, nearest, self.ef_construction, layer)
            chosen = self._select_neighbours(found, self.M)
            node.neighbours[layer] = chosen

            cap = self.max_m0 if layer == 0 else self.M
            for other_id in chosen:
                links = self.nodes[other_id].neighbours[layer]
                links.append(node_id)
                if len(links) > cap:  # too many links: re-pick the best `cap` of them
                    dists = self._dist_many(self._vectors[other_id], links).tolist()
                    ranked = sorted(zip(dists, links))
                    self.nodes[other_id].neighbours[layer] = self._select_neighbours(ranked, cap)
            nearest = found

        if level > self.max_level:
            self.entry_point, self.max_level = node_id, level
        return node_id

    def build(self, vectors: np.ndarray, document_ids: list[str]) -> None:
        if len(vectors) != len(document_ids):
            raise ValueError("vectors and document_ids must have the same length")
        for i, (vector, document_id) in enumerate(zip(vectors, document_ids), start=1):
            self.add(vector, document_id)
            if i % 5000 == 0:
                logger.info("inserted %d / %d vectors", i, len(document_ids))

    # ------------------------------------------------------------------ searching

    def search(self, query: np.ndarray, k: int = 10, ef_search: int | None = None) -> list[Neighbour]:
        """Approximate k nearest neighbours, nearest first."""
        if not self.nodes:
            return []
        if k <= 0:
            raise ValueError("k must be positive")
        ef = max(ef_search or self.ef_search, k)

        query = self._prepare(query)
        entry = self.entry_point
        nearest = [(float(self._dist_many(query, [entry])[0]), entry)]
        for layer in range(self.max_level, 0, -1):  # greedy descent
            nearest = self._search_layer(query, nearest, 1, layer)
        found = self._search_layer(query, nearest, ef, 0)[:k]  # wide search on layer 0
        return [Neighbour(self.nodes[i].document_id, dist) for dist, i in found]

    # ------------------------------------------------------------------ inspection

    def stats(self) -> dict:
        per_layer = [0] * (self.max_level + 1)
        for node in self.nodes:
            for layer in range(node.level + 1):
                per_layer[layer] += 1
        layer0_links = sum(len(n.neighbours[0]) for n in self.nodes)
        return {
            "nodes": len(self.nodes),
            "layers": self.max_level + 1,
            "nodes_per_layer": per_layer,
            "avg_layer0_degree": layer0_links / max(len(self.nodes), 1),
        }

    def memory_bytes(self) -> int:
        """Vector matrix plus the Python graph (list objects and their int entries)."""
        vectors = len(self.nodes) * self.dim * 4
        graph = 0
        for node in self.nodes:
            for links in node.neighbours:
                graph += sys.getsizeof(links) + 28 * len(links)
        return vectors + graph

    # ------------------------------------------------------------------ persistence

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "params": dict(
                dim=self.dim,
                max_elements=self.max_elements,
                M=self.M,
                ef_construction=self.ef_construction,
                ef_search=self.ef_search,
                metric=self.metric,
                seed=self.seed,
                use_heuristic=self.use_heuristic,
            ),
            "vectors": self._vectors[: len(self.nodes)],
            "document_ids": [n.document_id for n in self.nodes],
            "levels": [n.level for n in self.nodes],
            "neighbours": [n.neighbours for n in self.nodes],
            "entry_point": self.entry_point,
            "max_level": self.max_level,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "HNSWIndex":
        with open(path, "rb") as f:
            state = pickle.load(f)
        index = cls(**state["params"])
        n = len(state["document_ids"])
        index._vectors[:n] = state["vectors"]
        index.nodes = [
            Node(index._vectors[i], state["document_ids"][i], state["levels"][i], state["neighbours"][i])
            for i in range(n)
        ]
        index.entry_point = state["entry_point"]
        index.max_level = state["max_level"]
        return index
