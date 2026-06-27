"""Leave-one-out evaluation: Recall@K / HitRate@K vs a popularity baseline.

Protocol
--------
For every user with at least two ratings we hold out one *positively* rated item
(rating >= ``positive_threshold``) as the test target and train on the rest. A
recommender produces a top-K list from the user's remaining history; we score a
**hit** if the held-out item appears in that list.

* HitRate@K  = fraction of held-out users whose target is in the top-K.
* Recall@K   = hits / number of held-out targets. With exactly one held-out item
  per user this equals HitRate@K, but the metric is written for the general case.

The item-item similarity model is fitted on the *training* matrix only, so the
held-out interactions never leak into the neighbourhoods.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .data import Ratings


@dataclass
class LOOData:
    train: Ratings
    # per held-out user: (user_id, train_item_indices, train_ratings, target_item_index)
    queries: list


def leave_one_out_split(
    ratings: Ratings,
    positive_threshold: float = 4.0,
    min_history: int = 2,
    seed: int = 0,
) -> LOOData:
    """Hold out one positively-rated item per eligible user.

    Returns training Ratings (rebuilt without the held-out cells, on the SAME item
    index space) and a list of query tuples for scoring.
    """
    rng = random.Random(seed)
    R = ratings.matrix.tocsr()
    n_items = ratings.n_items

    train_triples = []
    queries = []

    for u in range(ratings.n_users):
        start, end = R.indptr[u], R.indptr[u + 1]
        items = R.indices[start:end]
        vals = R.data[start:end]
        if len(items) < min_history:
            train_triples.extend((u, int(i), float(v)) for i, v in zip(items, vals))
            continue

        positives = [j for j in range(len(items)) if vals[j] >= positive_threshold]
        if not positives or len(items) - 1 < 1:
            train_triples.extend((u, int(i), float(v)) for i, v in zip(items, vals))
            continue

        held = rng.choice(positives)
        target_item = int(items[held])
        kept_items, kept_vals = [], []
        for j in range(len(items)):
            if j == held:
                continue
            kept_items.append(int(items[j]))
            kept_vals.append(float(vals[j]))
            train_triples.append((u, int(items[j]), float(vals[j])))
        queries.append((u, np.asarray(kept_items), np.asarray(kept_vals), target_item))

    # rebuild train matrix on the SAME (user,item) index space so item indices align
    from scipy.sparse import csr_matrix

    if train_triples:
        rows = np.fromiter((t[0] for t in train_triples), dtype=np.int32, count=len(train_triples))
        cols = np.fromiter((t[1] for t in train_triples), dtype=np.int32, count=len(train_triples))
        data = np.fromiter((t[2] for t in train_triples), dtype=np.float64, count=len(train_triples))
        mat = csr_matrix((data, (rows, cols)), shape=(ratings.n_users, n_items))
    else:
        mat = ratings.matrix.copy()

    train = Ratings(mat, ratings.users, ratings.items, ratings.user_index, ratings.item_index)
    return LOOData(train=train, queries=queries)


def recall_at_k(hits: int, n_targets: int) -> float:
    return hits / n_targets if n_targets else 0.0


def hit_rate_at_k(hits: int, n_users: int) -> float:
    return hits / n_users if n_users else 0.0


def evaluate(model, loo: LOOData, k: int = 10) -> dict:
    """Run a fitted recommender over the held-out queries and report metrics.

    ``model`` must expose recommend_for_history(seen_idx, seen_vals, n) -> [item_index].
    """
    hits = 0
    n = len(loo.queries)
    for _u, seen_idx, seen_vals, target in loo.queries:
        recs = model.recommend_for_history(seen_idx, seen_vals, n=k)
        if target in recs:
            hits += 1
    return {
        "k": k,
        "n_users": n,
        "hits": hits,
        "recall_at_k": recall_at_k(hits, n),
        "hit_rate_at_k": hit_rate_at_k(hits, n),
    }
