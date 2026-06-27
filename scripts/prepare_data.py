"""Build a small, committable ratings slice from the Amazon Video Games 5-core corpus.

Source: McAuley/SNAP "Amazon product data" — Video Games 5-core
    https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz

The full file is ~113 MB compressed (~231k reviews) and far too large to commit.
This script streams it, keeps (user, item, rating) triples, iteratively k-cores the
graph so every user/item has enough co-ratings to be useful, and writes a compact
CSV to data/ratings.csv that the gamerec package and tests consume.

Usage:
    python scripts/prepare_data.py --src path/to/reviews_Video_Games_5.json.gz

If the source file is unavailable, fall back to scripts/make_synthetic.py instead.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "ratings.csv")


def stream_triples(src: str):
    op = gzip.open if src.endswith(".gz") else open
    with op(src, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            u = r.get("reviewerID")
            i = r.get("asin")
            rating = r.get("overall")
            t = r.get("unixReviewTime", 0)
            if u and i and rating is not None:
                yield u, i, float(rating), int(t or 0)


def kcore(rows, min_user=5, min_item=8, max_iter=20):
    """Iteratively drop users/items below the co-rating thresholds."""
    rows = list(rows)
    for _ in range(max_iter):
        uc = Counter(r[0] for r in rows)
        ic = Counter(r[1] for r in rows)
        keep = [r for r in rows if uc[r[0]] >= min_user and ic[r[1]] >= min_item]
        if len(keep) == len(rows):
            break
        rows = keep
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to reviews_Video_Games_5.json.gz")
    ap.add_argument("--min-user", type=int, default=5)
    ap.add_argument("--min-item", type=int, default=10)
    ap.add_argument("--max-items", type=int, default=1200,
                    help="cap catalogue to the most-rated items to keep the CSV small")
    args = ap.parse_args()

    print(f"streaming {args.src} ...")
    triples = list(stream_triples(args.src))
    print(f"  raw triples: {len(triples):,}")

    # de-duplicate (user,item): keep the latest review's rating
    latest: dict[tuple[str, str], tuple[float, int]] = {}
    for u, i, rating, t in triples:
        key = (u, i)
        if key not in latest or t >= latest[key][1]:
            latest[key] = (rating, t)
    rows = [(u, i, rv) for (u, i), (rv, _) in latest.items()]
    print(f"  unique (user,item): {len(rows):,}")

    # cap catalogue to the most-reviewed items (keeps the long head, drops noise)
    ic = Counter(i for _, i, _ in rows)
    top_items = {i for i, _ in ic.most_common(args.max_items)}
    rows = [r for r in rows if r[1] in top_items]
    print(f"  after item cap ({args.max_items}): {len(rows):,}")

    rows = kcore(rows, min_user=args.min_user, min_item=args.min_item)
    users = {r[0] for r in rows}
    items = {r[1] for r in rows}
    print(f"  after {args.min_user}/{args.min_item} k-core: "
          f"{len(rows):,} ratings, {len(users):,} users, {len(items):,} items")
    density = len(rows) / (len(users) * len(items)) if users and items else 0
    print(f"  density: {density:.4%}")

    # stable integer remapping for compact, anonymous IDs
    uidx = {u: k for k, u in enumerate(sorted(users))}
    iidx = {i: k for k, i in enumerate(sorted(items))}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user", "item", "rating"])
        for u, i, rv in sorted(rows, key=lambda r: (uidx[r[0]], iidx[r[1]])):
            w.writerow([uidx[u], iidx[i], f"{rv:.1f}"])
    print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
