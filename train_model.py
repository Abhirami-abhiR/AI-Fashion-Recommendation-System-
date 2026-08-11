"""
train_model.py
---------------
Trains the ML ranking model at the heart of the recommender.

Approach (hybrid):
  1. Content-based signal: adaptive-feature overlap between a user's
     disability-driven needs and each item (handled at inference time in
     recommender.py via cosine similarity).
  2. Learned ranking signal: a RandomForestRegressor trained on historical /
     simulated (user, item) -> suitability_rating interactions. This lets the
     model pick up on patterns beyond simple overlap (e.g. price/budget fit,
     style match, interaction effects between features).

The final recommendation score blends both signals (see recommender.py).

Run:
    python train_model.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import sys

sys.path.append(str(Path(__file__).parent))
from preprocessing import FeatureEncoder

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    users = pd.read_csv(DATA_DIR / "users.csv")
    items = pd.read_csv(DATA_DIR / "clothing_items.csv")
    interactions = pd.read_csv(DATA_DIR / "interactions.csv")

    encoder = FeatureEncoder().fit(users, items)
    users_enc = encoder.transform_users(users)
    items_enc = encoder.transform_items(items)

    # Build the full training matrix by joining interactions with encoded features
    df = interactions.merge(users_enc, on="user_id").merge(
        items_enc, on="item_id", suffixes=("_user", "_item")
    )

    feature_cols = [c for c in df.columns if c not in
                    ("user_id", "item_id", "suitability_rating")]
    X = df[feature_cols]
    y = df["suitability_rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MAE:  {mae:.3f}  (rating scale 1-5)")
    print(f"Test R^2:  {r2:.3f}")

    # Feature importance (top 10) — useful for the report / README
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top10 = importances.sort_values(ascending=False).head(10)
    print("\nTop 10 most important features:")
    print(top10.to_string())

    joblib.dump(model, MODEL_DIR / "ranking_model.pkl")
    encoder.save(str(MODEL_DIR / "feature_encoder.pkl"))
    joblib.dump(feature_cols, MODEL_DIR / "feature_columns.pkl")

    print(f"\nSaved model      -> {MODEL_DIR / 'ranking_model.pkl'}")
    print(f"Saved encoder    -> {MODEL_DIR / 'feature_encoder.pkl'}")
    print(f"Saved feat. cols -> {MODEL_DIR / 'feature_columns.pkl'}")


if __name__ == "__main__":
    main()
