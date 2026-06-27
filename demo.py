"""Demo: real recommendations for a sample user + held-out evaluation.

Runs on data/ratings.csv (the Amazon Video Games 5-core slice). Prints a sample
user's top games, an item-to-item "if you liked X" example, and Recall@K /
HitRate@K for item-based CF vs a popularity baseline.

    python demo.py
"""
from __future__ import annotations

import os

from gamerec import (
    Ratings,
    ItemCF,
    PopularityRecommender,
    leave_one_out_split,
    evaluate,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "ratings.csv")
K = 10


def main():
    print("=" * 64)
    print("game-recommender - item-based collaborative filtering demo")
    print("=" * 64)

    ratings = Ratings.from_csv(DATA)
    print(f"\nDataset: {DATA}")
    print(f"  {ratings.nnz:,} ratings  |  {ratings.n_users:,} users  |  "
          f"{ratings.n_items:,} games  |  density {ratings.density:.3%}")

    # ---- recommendations for a sample user ----------------------------------
    model = ItemCF(k=40, min_overlap=3, shrink=10.0).fit(ratings)

    # pick a user with a reasonably long, opinionated history
    best_u, best_len = None, 0
    for u in ratings.users:
        idx, _ = ratings.user_row(u)
        if len(idx) > best_len:
            best_len, best_u = len(idx), u
    sample_user = best_u

    seen_idx, seen_vals = ratings.user_row(sample_user)
    print(f"\nSample user {sample_user} has rated {len(seen_idx)} games.")
    top_seen = sorted(zip(seen_idx, seen_vals), key=lambda x: -x[1])[:5]
    print("  Top-rated games (item_id : rating):")
    for i, v in top_seen:
        print(f"    game {ratings.items[i]:>10}  :  {v:.0f}")

    recs = model.recommend(ratings, sample_user, n=K)
    print(f"\n  Top-{K} recommendations (item_id : score):")
    for i, s in recs:
        print(f"    game {ratings.items[i]:>10}  :  {s:6.3f}")

    # ---- "if you liked this game" item-to-item example ----------------------
    seed_item = top_seen[0][0]
    print(f"\n  Games most similar to game {ratings.items[seed_item]} "
          f"(item_id : similarity):")
    for i, sim in model.similar_items(seed_item, n=5):
        print(f"    game {ratings.items[i]:>10}  :  {sim:6.3f}")

    # ---- held-out evaluation: CF vs popularity ------------------------------
    print("\n" + "-" * 64)
    print(f"Leave-one-out evaluation (hold out one 4+ star game / user), K={K}")
    print("-" * 64)

    loo = leave_one_out_split(ratings, positive_threshold=4.0, seed=0)
    cf = ItemCF(k=40, min_overlap=3, shrink=10.0).fit(loo.train)
    pop = PopularityRecommender().fit(loo.train)

    cf_m = evaluate(cf, loo, k=K)
    pop_m = evaluate(pop, loo, k=K)

    print(f"  held-out users: {cf_m['n_users']:,}")
    print(f"  {'model':<22}{'Recall@'+str(K):>12}{'HitRate@'+str(K):>14}{'hits':>8}")
    print(f"  {'item-based CF':<22}{cf_m['recall_at_k']:>12.4f}"
          f"{cf_m['hit_rate_at_k']:>14.4f}{cf_m['hits']:>8}")
    print(f"  {'popularity baseline':<22}{pop_m['recall_at_k']:>12.4f}"
          f"{pop_m['hit_rate_at_k']:>14.4f}{pop_m['hits']:>8}")
    lift = (cf_m["recall_at_k"] / pop_m["recall_at_k"] - 1) * 100 if pop_m["recall_at_k"] else float("inf")
    print(f"\n  CF lift over popularity: +{lift:.1f}% Recall@{K}")


if __name__ == "__main__":
    main()
