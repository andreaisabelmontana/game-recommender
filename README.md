# Game Recommender

A video-game recommender trained on Amazon review signals — item-based collaborative
filtering that turns sparse, skewed review data into "if you liked this, play that"
suggestions.

- **Live site:** https://andreaisabelmontana.github.io/game-recommender/

## Data

Real data: the **Amazon Video Games 5-core** review corpus (McAuley / SNAP,
`reviews_Video_Games_5.json.gz`, ~231k reviews). The full file is ~113 MB compressed,
so `scripts/prepare_data.py` streams it, de-duplicates `(user, item)` pairs, caps the
catalogue to the most-reviewed titles, and iteratively applies a 5/10 k-core (every
user keeps ≥5 ratings, every game ≥10) to remove the sparsest noise. The committed
slice is `data/ratings.csv`:

| ratings | users | games | density |
|--------:|------:|------:|--------:|
| 71,746  | 8,314 | 1,198 | 0.720%  |

IDs are remapped to compact anonymous integers. A synthetic generator with planted
genre structure (`scripts/make_synthetic.py`) is also included — used as a controlled
test fixture and as a fallback if the corpus can't be downloaded.

## Method — item-based collaborative filtering

Let `r[u,i]` be user *u*'s rating of game *i*, and `mu[u]` their mean rating.
Mean-centring each user row turns plain cosine into **adjusted cosine** similarity,
which cancels the "some people rate everything 5 stars" bias:

```
s(i,j) =        sum_u (r[u,i]-mu[u]) (r[u,j]-mu[u])
         ----------------------------------------------------
         ||r[.,i]-mu||  ||r[.,j]-mu||      (over users who rated both)
```

Two sparsity defences:

- **Minimum overlap** — a similarity built on 1–2 shared raters is noise; pairs with
  fewer than `min_overlap` co-raters are dropped.
- **Significance shrinkage** — `s'(i,j) = s(i,j) · n_ij / (n_ij + shrink)`, where
  `n_ij` is the number of co-raters. Thinly-supported pairs are pulled toward 0;
  well-supported ones are barely touched.

The diagonal is zeroed and only the top-`k` neighbours per game are kept. A user's
score for an unseen game *j* is the similarity-weighted sum of their mean-centred
ratings over *j*'s neighbours they have actually played; already-rated games are
excluded from the top-N. (The adjusted-cosine computation is cross-checked against
`sklearn.metrics.pairwise.cosine_similarity` in the test suite.)

## Evaluation

**Leave-one-out**: for every user, hold out one 4★+ game as the test target, fit
item-item similarity on the *training* matrix only (no leakage), and check whether the
held-out game appears in the model's top-K. With one target per user, Recall@K equals
HitRate@K. Baseline = recommend the globally most-rated games.

Real numbers from `python demo.py` on `data/ratings.csv` (8,215 held-out users):

| K  | item-based CF | popularity | lift |
|---:|-------------:|-----------:|-----:|
| 5  | 0.0668       | 0.0348     | +92.0% |
| 10 | 0.1042       | 0.0540     | +92.8% |
| 20 | 0.1529       | 0.0857     | +78.4% |

At K=10 the recommender finds the held-out game for **856 / 8,215** users (Recall@10
0.1042) versus **444** for the popularity baseline — a **+92.8%** lift.

## Run it

```bash
pip install -r requirements.txt

# regenerate data/ratings.csv from the raw corpus (optional — the slice is committed)
python scripts/prepare_data.py --src reviews_Video_Games_5.json.gz

python demo.py          # recommendations for a sample user + eval metrics
python -m pytest -q     # 8 tests
```

## Layout

```
gamerec/        package: data loading, item-based CF model, evaluation
  data.py       sparse user x item matrix + index maps
  model.py      adjusted-cosine similarity, shrinkage, top-N recommend
  evaluate.py   leave-one-out split, Recall@K / HitRate@K
scripts/        prepare_data.py (real corpus) + make_synthetic.py (fixture/fallback)
tests/          pytest suite
data/           committed ratings slice
demo.py
```

## Tests

`python -m pytest -q` — 8 tests covering: identical co-rated items rank as nearest
neighbours; recommendations exclude already-seen games; shrinkage down-weights
low-overlap pairs; min-overlap filtering; adjusted cosine matches scikit-learn; and
Recall@K beating the popularity baseline on data with planted genre structure.

Built from scratch as a learning project.
