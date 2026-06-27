"""Item-based collaborative filtering with adjusted-cosine similarity.

Math
----
Let r[u,i] be user u's rating of item i. Mean-centring each *user* row by their
average rating mu[u] removes the "some people rate everything high" bias and turns
plain cosine into **adjusted cosine** similarity:

    s(i, j) = sum_u (r[u,i] - mu[u]) * (r[u,j] - mu[u])
              ---------------------------------------------
              ||r[.,i] - mu||  *  ||r[.,j] - mu||

computed only over users who rated *both* i and j. Two failure modes are handled:

* **Sparsity / small overlap.** A similarity built on 1-2 shared raters is noise.
  We require ``min_overlap`` co-ratings and apply *significance shrinkage*:

      s'(i,j) = s(i,j) * n_ij / (n_ij + shrink)

  where n_ij is the number of users who rated both. Pairs with few co-raters get
  pulled toward 0; well-supported pairs are barely touched.

* **Self / popularity.** Diagonal is zeroed; only the top ``k`` neighbours per item
  are kept so a query aggregates over genuine neighbours, not the whole catalogue.

Scoring
-------
A user u's predicted affinity for an unseen item j is the similarity-weighted sum of
their mean-centred ratings over j's neighbours that u has actually rated:

    score(u, j) = sum_{i in N(j) ∩ seen(u)} s'(i,j) * (r[u,i] - mu[u])

Top-N recommendations are the highest-scoring items u has not already rated.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from .data import Ratings


class ItemCF:
    def __init__(self, k: int = 40, min_overlap: int = 3, shrink: float = 10.0):
        """
        Args:
            k:           neighbours kept per item.
            min_overlap: minimum co-raters for a pair to get a non-zero similarity.
            shrink:      significance-weighting term; larger = more aggressive
                         down-weighting of low-overlap pairs.
        """
        self.k = k
        self.min_overlap = min_overlap
        self.shrink = shrink
        self.sim_: csr_matrix | None = None
        self.user_means_: np.ndarray | None = None
        self.train_: csr_matrix | None = None

    def fit(self, ratings: Ratings) -> "ItemCF":
        R = ratings.matrix.tocsr().astype(np.float64)
        n_users, n_items = R.shape
        self.train_ = R

        # per-user mean over observed ratings only
        sums = np.asarray(R.sum(axis=1)).ravel()
        counts = np.diff(R.indptr)
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        self.user_means_ = means

        # mean-centre each user's observed entries (keep sparsity: only nnz cells)
        C = R.copy()
        C.data = C.data - np.repeat(means, counts)

        # column-major view: each item's centred rating vector
        Cc = C.tocsc()

        # numerator: Gram matrix of centred item vectors (items x items)
        num = (Cc.T @ Cc).tocsr()

        # overlap counts: how many users rated both i and j.
        # B is the binary users x items matrix (1 where rated); B.T @ B counts,
        # for each item pair, the users who rated both.
        B = csr_matrix(
            (np.ones_like(R.data), R.indices, R.indptr), shape=R.shape
        )
        overlap = (B.T @ B).tocsr()

        # item norms (sqrt of diagonal of num)
        diag = num.diagonal().copy()
        norms = np.sqrt(np.maximum(diag, 0.0))

        sim = num.tocoo()
        rows, cols, vals = sim.row, sim.col, sim.data

        # divide by norm_i * norm_j
        denom = norms[rows] * norms[cols]
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = np.where(denom > 0, vals / denom, 0.0)

        # overlap-based filtering + significance shrinkage
        ov = np.asarray(overlap[rows, cols]).ravel()
        cos = np.where(ov >= self.min_overlap, cos, 0.0)
        cos = cos * (ov / (ov + self.shrink))

        # drop diagonal and zeros
        keep = (rows != cols) & (cos != 0.0)
        rows, cols, cos = rows[keep], cols[keep], cos[keep]

        full = csr_matrix((cos, (rows, cols)), shape=(n_items, n_items))
        self.sim_ = self._topk_per_row(full, self.k)
        return self

    @staticmethod
    def _topk_per_row(S: csr_matrix, k: int) -> csr_matrix:
        """Keep only the k largest entries in each row."""
        S = S.tocsr()
        data, indices, indptr = [], [], [0]
        for r in range(S.shape[0]):
            start, end = S.indptr[r], S.indptr[r + 1]
            d = S.data[start:end]
            idx = S.indices[start:end]
            if len(d) > k:
                top = np.argpartition(d, -k)[-k:]
                d, idx = d[top], idx[top]
            order = np.argsort(-d)
            data.extend(d[order])
            indices.extend(idx[order])
            indptr.append(len(data))
        return csr_matrix(
            (np.asarray(data), np.asarray(indices, dtype=np.int32),
             np.asarray(indptr, dtype=np.int32)),
            shape=S.shape,
        )

    def similar_items(self, item_index: int, n: int = 10):
        """Return [(item_index, similarity)] nearest neighbours of an item."""
        if self.sim_ is None:
            raise RuntimeError("model not fitted")
        start, end = self.sim_.indptr[item_index], self.sim_.indptr[item_index + 1]
        idx = self.sim_.indices[start:end]
        val = self.sim_.data[start:end]
        order = np.argsort(-val)[:n]
        return list(zip(idx[order].tolist(), val[order].tolist()))

    def _scores_for_seen(self, seen_idx: np.ndarray, seen_centred: np.ndarray) -> np.ndarray:
        """Aggregate neighbour similarities over a user's seen items -> item scores."""
        n_items = self.sim_.shape[0]
        scores = np.zeros(n_items, dtype=np.float64)
        for i, cr in zip(seen_idx, seen_centred):
            start, end = self.sim_.indptr[i], self.sim_.indptr[i + 1]
            nbr = self.sim_.indices[start:end]
            sval = self.sim_.data[start:end]
            scores[nbr] += sval * cr
        scores[seen_idx] = -np.inf  # never recommend an already-seen item
        return scores

    def recommend(self, ratings: Ratings, user_id, n: int = 10):
        """Top-N (item_index, score) recommendations for an original user id."""
        u = ratings.user_index[user_id]
        mu = self.user_means_[u] if u < len(self.user_means_) else 0.0
        seen_idx, seen_vals = ratings.user_row(user_id)
        centred = seen_vals - mu
        scores = self._scores_for_seen(np.asarray(seen_idx), np.asarray(centred))
        if not np.any(np.isfinite(scores) & (scores > 0)):
            return []
        order = np.argsort(-scores)[:n]
        return [(int(j), float(scores[j])) for j in order if np.isfinite(scores[j]) and scores[j] > 0]

    def recommend_for_history(self, seen_idx, seen_vals, n: int = 10):
        """Top-N for an ad-hoc history (used by leave-one-out evaluation).

        seen_vals are raw ratings; they are mean-centred by the history's own mean.
        """
        seen_idx = np.asarray(seen_idx, dtype=np.int64)
        seen_vals = np.asarray(seen_vals, dtype=np.float64)
        mu = seen_vals.mean() if len(seen_vals) else 0.0
        scores = self._scores_for_seen(seen_idx, seen_vals - mu)
        order = np.argsort(-scores)[:n]
        return [int(j) for j in order if np.isfinite(scores[j]) and scores[j] > 0]


class PopularityRecommender:
    """Non-personalized baseline: recommend the globally most-rated items."""

    def __init__(self):
        self.ranking_: np.ndarray | None = None

    def fit(self, ratings: Ratings) -> "PopularityRecommender":
        counts = np.diff(ratings.matrix.tocsc().indptr)
        self.ranking_ = np.argsort(-counts)
        return self

    def recommend_for_history(self, seen_idx, seen_vals=None, n: int = 10):
        seen = set(int(i) for i in seen_idx)
        out = []
        for j in self.ranking_:
            j = int(j)
            if j in seen:
                continue
            out.append(j)
            if len(out) >= n:
                break
        return out
