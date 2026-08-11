"""
evaluate.py
-----------
Offline evaluation of the trained ranking model + a simple ranking-quality
check (Precision@K using a relevance threshold on simulated ratings).

Run:
    python evaluate.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).parent))
from recommender import AdaptiveFashionRecommender

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"

RELEVANCE_THRESHOLD = 3.5  # ratings >= this are considered "relevant" for Precision@K


def precision_at_k(rec: AdaptiveFashionRecommender, users: pd.DataFrame,
                    interactions: pd.DataFrame, k: int = 5) -> float:
    precisions = []
    for _, u in users.sample(n=min(60, len(users)), random_state=1).iterrows():
        user_interactions = interactions[interactions["user_id"] == u["user_id"]]
        if user_interactions.empty:
            continue
        relevant_items = set(
            user_interactions[user_interactions["suitability_rating"] >= RELEVANCE_THRESHOLD]["item_id"]
        )
        if not relevant_items:
            continue
        recs = rec.recommend(
            disability_type=u["disability_type"],
            preferred_style=u["preferred_style"],
            budget_tier=u["budget_tier"],
            age=int(u["age"]),
            top_k=k,
        )
        recommended_ids = set(recs["item_id"])
        hit = len(recommended_ids & relevant_items)
        precisions.append(hit / k)
    return float(np.mean(precisions)) if precisions else 0.0


def main():
    users = pd.read_csv(DATA_DIR / "users.csv")
    interactions = pd.read_csv(DATA_DIR / "interactions.csv")

    rec = AdaptiveFashionRecommender(model_dir=str(MODEL_DIR), data_dir=str(DATA_DIR))

    p_at_5 = precision_at_k(rec, users, interactions, k=5)
    p_at_10 = precision_at_k(rec, users, interactions, k=10)

    print("Offline evaluation (sampled users)")
    print(f"  Precision@5:  {p_at_5:.3f}")
    print(f"  Precision@10: {p_at_10:.3f}")
    print(f"  (Relevance threshold: simulated rating >= {RELEVANCE_THRESHOLD})")


if __name__ == "__main__":
    main()
