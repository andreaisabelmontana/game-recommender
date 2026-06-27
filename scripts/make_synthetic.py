"""Generate a synthetic user x game ratings matrix with planted genre structure.

Used as (a) a fallback if the real Amazon corpus can't be downloaded, and (b) a
controlled fixture for tests: because preferences are driven by latent genres,
item-based CF should recover same-genre neighbours and beat a popularity baseline.

Each user is assigned an affinity over G latent genres; each game belongs to one
genre. A user's rating for a game is their genre affinity plus noise, observed only
for a sparse random subset of games (with popularity-skewed exposure so a few games
get many ratings — mimicking the long tail).
"""
from __future__ import annotations

import numpy as np

from gamerec.data import Ratings


def make_synthetic(
    n_users: int = 800,
    n_items: int = 200,
    n_genres: int = 8,
    obs_per_user: int = 25,
    noise: float = 0.6,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)

    item_genre = rng.integers(0, n_genres, size=n_items)
    # popularity-skewed exposure (long tail)
    pop = rng.zipf(1.6, size=n_items).astype(float)
    pop = pop / pop.sum()

    # each user likes 1-2 genres strongly
    user_aff = rng.uniform(1.0, 2.0, size=(n_users, n_genres))
    for u in range(n_users):
        liked = rng.choice(n_genres, size=rng.integers(1, 3), replace=False)
        user_aff[u, liked] += rng.uniform(2.5, 3.5, size=len(liked))

    triples = []
    for u in range(n_users):
        chosen = rng.choice(n_items, size=min(obs_per_user, n_items), replace=False, p=pop)
        for i in chosen:
            base = user_aff[u, item_genre[i]]
            r = base + rng.normal(0, noise)
            r = float(np.clip(round(r), 1, 5))
            triples.append((f"u{u}", f"g{i}", r))

    ratings = Ratings.from_triples(triples)
    return ratings, item_genre


if __name__ == "__main__":
    import csv
    import os

    ratings, _ = make_synthetic()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "data", "synthetic_ratings.csv")
    R = ratings.matrix.tocoo()
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user", "item", "rating"])
        for u, i, r in zip(R.row, R.col, R.data):
            w.writerow([int(u), int(i), f"{r:.1f}"])
    print(f"wrote {out}: {ratings.nnz} ratings, "
          f"{ratings.n_users} users, {ratings.n_items} items, "
          f"density {ratings.density:.2%}")
