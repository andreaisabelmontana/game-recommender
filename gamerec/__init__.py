"""gamerec — item-based collaborative filtering for video-game recommendations.

Public API:
    Ratings            — load a (user, item, rating) table into a sparse matrix
    ItemCF             — fit item-item similarity and generate top-N recommendations
    PopularityRecommender — non-personalized popularity baseline
    leave_one_out_split, evaluate, recall_at_k, hit_rate_at_k — evaluation utilities
"""
from .data import Ratings
from .model import ItemCF, PopularityRecommender
from .evaluate import (
    leave_one_out_split,
    evaluate,
    recall_at_k,
    hit_rate_at_k,
)

__all__ = [
    "Ratings",
    "ItemCF",
    "PopularityRecommender",
    "leave_one_out_split",
    "evaluate",
    "recall_at_k",
    "hit_rate_at_k",
]

__version__ = "0.1.0"
