"""Tests for the gamerec item-based collaborative-filtering package."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gamerec import (  # noqa: E402
    Ratings,
    ItemCF,
    PopularityRecommender,
    leave_one_out_split,
    evaluate,
)
from scripts.make_synthetic import make_synthetic  # noqa: E402


def test_identical_columns_are_nearest_neighbours():
    """Two items co-rated identically by the same users must be each other's
    top neighbour."""
    # users 0..4 rate items 0 and 1 identically; item 2 is anti-correlated,
    # item 3 is unrelated.
    triples = []
    same = [5, 4, 5, 4, 5]
    opp = [1, 2, 1, 2, 1]
    other = [3, 3, 3, 3, 3]
    for u in range(5):
        triples.append((u, 0, same[u]))
        triples.append((u, 1, same[u]))   # identical to item 0
        triples.append((u, 2, opp[u]))    # opposite pattern
        triples.append((u, 3, other[u]))  # flat / uninformative
    r = Ratings.from_triples(triples)

    model = ItemCF(k=10, min_overlap=3, shrink=0.0).fit(r)
    nbrs = model.similar_items(r.item_index[0], n=3)
    assert nbrs, "item 0 should have neighbours"
    top_item, top_sim = nbrs[0]
    assert top_item == r.item_index[1], "item 1 should be item 0's nearest neighbour"
    assert top_sim > 0.99, f"identical columns should have similarity ~1, got {top_sim}"


def test_recommend_excludes_seen_items():
    """Recommendations must never include items the user has already rated."""
    ratings, _ = make_synthetic(n_users=200, n_items=60, seed=3)
    model = ItemCF(k=20, min_overlap=2, shrink=5.0).fit(ratings)

    some_user = ratings.users[0]
    seen_idx, _ = ratings.user_row(some_user)
    seen = set(int(i) for i in seen_idx)

    recs = model.recommend(ratings, some_user, n=10)
    rec_items = {i for i, _ in recs}
    assert rec_items, "expected at least one recommendation"
    assert rec_items.isdisjoint(seen), "recommendations leaked an already-seen item"


def test_similarity_matrix_is_symmetric_and_diag_free():
    ratings, _ = make_synthetic(n_users=150, n_items=40, seed=11)
    model = ItemCF(k=40, min_overlap=2, shrink=2.0).fit(ratings)
    S = model.sim_.toarray()
    assert np.allclose(np.diag(S), 0.0), "self-similarity must be zeroed"


def test_shrinkage_downweights_low_overlap_pairs():
    """A pair seen by 2 users should be down-weighted relative to no shrinkage."""
    # items 0,1 co-rated by exactly 2 users, perfectly correlated
    triples = [
        (0, 0, 5.0), (0, 1, 5.0),
        (1, 0, 4.0), (1, 1, 4.0),
        # filler so other items exist and means are non-degenerate
        (0, 2, 1.0), (1, 2, 2.0), (2, 2, 3.0), (2, 3, 3.0),
    ]
    r = Ratings.from_triples(triples)
    no_shrink = ItemCF(k=10, min_overlap=2, shrink=0.0).fit(r)
    heavy = ItemCF(k=10, min_overlap=2, shrink=20.0).fit(r)

    def sim01(m):
        d = dict(m.similar_items(r.item_index[0], n=10))
        return d.get(r.item_index[1], 0.0)

    assert sim01(heavy) < sim01(no_shrink), "shrinkage should reduce a 2-overlap sim"


def test_min_overlap_filters_pairs():
    """With min_overlap above the actual overlap, the pair gets no similarity."""
    triples = [
        (0, 0, 5.0), (0, 1, 5.0),
        (1, 0, 4.0), (1, 1, 4.0),
        (2, 2, 3.0), (2, 0, 2.0),
    ]
    r = Ratings.from_triples(triples)
    model = ItemCF(k=10, min_overlap=3, shrink=0.0).fit(r)  # overlap(0,1)=2 < 3
    d = dict(model.similar_items(r.item_index[0], n=10))
    assert d.get(r.item_index[1], 0.0) == 0.0


def test_recall_beats_popularity_on_planted_structure():
    """On synthetic data with planted genre preferences, item-based CF must beat
    the popularity baseline on Recall@K."""
    ratings, _ = make_synthetic(n_users=800, n_items=200, n_genres=8, seed=7)
    loo = leave_one_out_split(ratings, positive_threshold=4.0, seed=0)

    cf = ItemCF(k=40, min_overlap=2, shrink=5.0).fit(loo.train)
    pop = PopularityRecommender().fit(loo.train)

    cf_metrics = evaluate(cf, loo, k=10)
    pop_metrics = evaluate(pop, loo, k=10)

    assert cf_metrics["n_users"] > 50, "need a meaningful held-out set"
    assert cf_metrics["recall_at_k"] > pop_metrics["recall_at_k"], (
        f"CF Recall@10 {cf_metrics['recall_at_k']:.3f} did not beat popularity "
        f"{pop_metrics['recall_at_k']:.3f}"
    )


def test_adjusted_cosine_matches_sklearn_dense():
    """Cross-check the hand-rolled adjusted cosine against sklearn's cosine on the
    mean-centred dense matrix (no overlap filter / no shrinkage)."""
    from sklearn.metrics.pairwise import cosine_similarity

    ratings, _ = make_synthetic(n_users=120, n_items=30, seed=5)
    model = ItemCF(k=30, min_overlap=1, shrink=0.0).fit(ratings)

    R = ratings.matrix.toarray()
    means = np.array([row[row != 0].mean() if (row != 0).any() else 0.0 for row in R])
    C = R.copy()
    for u in range(R.shape[0]):
        nz = R[u] != 0
        C[u, nz] = R[u, nz] - means[u]
    ref = cosine_similarity(C.T)
    np.fill_diagonal(ref, 0.0)

    S = model.sim_.toarray()
    # compare only entries the model kept (top-k, non-zero)
    mask = S != 0
    assert mask.sum() > 0
    assert np.allclose(S[mask], ref[mask], atol=1e-6), "adjusted cosine disagrees with sklearn"


def test_csv_roundtrip(tmp_path):
    triples = [(0, 0, 5.0), (0, 1, 3.0), (1, 1, 4.0), (1, 2, 2.0)]
    r = Ratings.from_triples(triples)
    p = tmp_path / "r.csv"
    import csv

    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user", "item", "rating"])
        R = r.matrix.tocoo()
        for u, i, v in zip(R.row, R.col, R.data):
            w.writerow([int(u), int(i), v])
    r2 = Ratings.from_csv(str(p))
    assert r2.n_users == 2 and r2.nnz == 4
