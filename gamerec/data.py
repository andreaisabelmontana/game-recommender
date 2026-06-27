"""Load a (user, item, rating) table into a sparse CSR user x item matrix."""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class Ratings:
    """A user x item ratings matrix plus the index maps to recover original IDs.

    Attributes:
        matrix:  CSR sparse matrix of shape (n_users, n_items); 0 = unobserved.
        users:   list mapping internal row index -> original user id.
        items:   list mapping internal col index -> original item id.
        user_index / item_index: original id -> internal index.
    """

    matrix: csr_matrix
    users: list
    items: list
    user_index: dict
    item_index: dict

    @property
    def n_users(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_items(self) -> int:
        return self.matrix.shape[1]

    @property
    def nnz(self) -> int:
        return self.matrix.nnz

    @property
    def density(self) -> float:
        rows, cols = self.matrix.shape
        return self.matrix.nnz / (rows * cols) if rows and cols else 0.0

    def user_row(self, user_id):
        """Return (item_indices, ratings) the user has rated, by internal index."""
        r = self.user_index[user_id]
        start, end = self.matrix.indptr[r], self.matrix.indptr[r + 1]
        return self.matrix.indices[start:end], self.matrix.data[start:end]

    @classmethod
    def from_triples(cls, triples) -> "Ratings":
        """Build from an iterable of (user, item, rating).

        Duplicate (user, item) pairs keep the last rating seen.
        """
        cell: dict[tuple, float] = {}
        users: list = []
        items: list = []
        uidx: dict = {}
        iidx: dict = {}
        for u, i, r in triples:
            if u not in uidx:
                uidx[u] = len(users)
                users.append(u)
            if i not in iidx:
                iidx[i] = len(items)
                items.append(i)
            cell[(uidx[u], iidx[i])] = float(r)

        if not cell:
            raise ValueError("no ratings provided")

        rows = np.fromiter((k[0] for k in cell), dtype=np.int32, count=len(cell))
        cols = np.fromiter((k[1] for k in cell), dtype=np.int32, count=len(cell))
        data = np.fromiter(cell.values(), dtype=np.float64, count=len(cell))
        mat = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        mat.sum_duplicates()
        return cls(mat, users, items, uidx, iidx)

    @classmethod
    def from_csv(cls, path: str) -> "Ratings":
        """Load a CSV with header columns user,item,rating."""
        def gen():
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield row["user"], row["item"], float(row["rating"])

        return cls.from_triples(gen())
