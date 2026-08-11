"""
recommender.py
---------------
The AdaptiveFashionRecommender class: a hybrid recommendation engine.

Final score for a (user, item) pair =
    ALPHA * content_similarity  +  (1 - ALPHA) * ml_predicted_rating_normalized

- content_similarity: cosine similarity between the user's disability-driven
  "needs vector" and the item's adaptive-feature vector. Transparent and
  directly explainable to end users / evaluators ("recommended because it has
  a magnetic closure and elastic waist, matching your profile").
- ml_predicted_rating: output of the trained RandomForestRegressor, which
  captures learned patterns (style match, price/budget fit, feature
  interactions) beyond simple overlap.

This hybrid design is a standard approach in recommender systems: content-based
filtering solves the cold-start problem (works for brand-new users/items) while
the learned model improves ranking quality using historical interaction data.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import sys

sys.path.append(str(Path(__file__).parent))
from preprocessing import FeatureEncoder, ADAPTIVE_FEATURES

DISABILITY_FEATURE_MAP = {
    "Wheelchair user": ["side_seated_fit", "wheelchair_back_length", "adjustable_hem", "elastic_waist", "front_opening"],
    "Limited hand dexterity": ["magnetic_closure", "velcro_closure", "no_buttons", "one_handed_zipper_pull", "elastic_waist"],
    "Visual impairment": ["high_contrast_pattern", "simple_texture_id", "tagless"],
    "Sensory sensitivity": ["tagless", "seamless_interior", "simple_texture_id"],
    "Limb difference (upper)": ["magnetic_closure", "one_handed_zipper_pull", "wide_arm_hole", "removable_sleeve", "front_opening"],
    "Limb difference (lower)": ["extra_wide_leg_opening", "adjustable_hem", "elastic_waist"],
    "Post-stroke / hemiplegia": ["front_opening", "magnetic_closure", "wide_arm_hole", "no_buttons"],
    "Amputee (prosthetic user)": ["prosthetic_leg_access_zip", "extra_wide_leg_opening", "adjustable_hem"],
    "Dwarfism / short stature": ["adjustable_hem", "elastic_waist"],
    "None / general accessibility": [],
}

ALPHA = 0.5  # blend weight between content-based similarity and ML model score


class AdaptiveFashionRecommender:
    def __init__(self, model_dir: str = "../models", data_dir: str = "../data"):
        model_dir = Path(model_dir)
        data_dir = Path(data_dir)

        self.model = joblib.load(model_dir / "ranking_model.pkl")
        self.encoder: FeatureEncoder = FeatureEncoder.load(str(model_dir / "feature_encoder.pkl"))
        self.feature_cols = joblib.load(model_dir / "feature_columns.pkl")

        self.items = pd.read_csv(data_dir / "clothing_items.csv")
        self.items_enc = self.encoder.transform_items(self.items)

    # ------------------------------------------------------------------ #
    def _needs_vector(self, disability_type: str) -> np.ndarray:
        relevant = DISABILITY_FEATURE_MAP.get(disability_type, [])
        return np.array([1 if f in relevant else 0 for f in ADAPTIVE_FEATURES]).reshape(1, -1)

    # ------------------------------------------------------------------ #
    def _content_scores(self, disability_type: str) -> np.ndarray:
        needs_vec = self._needs_vector(disability_type)
        item_feat_matrix = self.items_enc[ADAPTIVE_FEATURES].values
        if needs_vec.sum() == 0:
            # No specific disability profile -> neutral score for everyone
            return np.full(len(self.items_enc), 0.5)
        sims = cosine_similarity(needs_vec, item_feat_matrix).flatten()
        return sims

    # ------------------------------------------------------------------ #
    def _ml_scores(self, user_row: dict) -> np.ndarray:
        user_df = pd.DataFrame([{
            "user_id": 0,
            "disability_type": user_row["disability_type"],
            "preferred_style": user_row["preferred_style"],
            "budget_tier": user_row["budget_tier"],
            "age": user_row["age"],
        }])
        user_enc = self.encoder.transform_users(user_df)
        pair_features = self.encoder.build_pair_features(user_enc, self.items_enc)
        pair_features = pair_features.reindex(columns=self.feature_cols, fill_value=0)
        preds = self.model.predict(pair_features)
        # normalize 1-5 rating scale to 0-1
        return (preds - 1) / 4

    # ------------------------------------------------------------------ #
    def recommend(self, disability_type: str, preferred_style: str = "Casual",
                   budget_tier: str = "Medium", age: int = 30,
                   category: str | None = None, top_k: int = 10,
                   alpha: float = ALPHA) -> pd.DataFrame:
        """Returns the top_k recommended items for a user profile, with an
        explanation of which adaptive features drove the match."""
        user_row = dict(disability_type=disability_type, preferred_style=preferred_style,
                         budget_tier=budget_tier, age=age)

        content = self._content_scores(disability_type)
        ml = self._ml_scores(user_row)
        final_score = alpha * content + (1 - alpha) * ml

        result = self.items.copy()
        result["content_score"] = content
        result["ml_score"] = ml
        result["final_score"] = final_score

        if category:
            result = result[result["category"] == category]

        relevant_feats = set(DISABILITY_FEATURE_MAP.get(disability_type, []))

        def explain(row):
            if row["adaptive_features"] == "none":
                item_feats = set()
            else:
                item_feats = set(row["adaptive_features"].split("|"))
            matched = item_feats & relevant_feats
            if matched:
                return "Matches your needs: " + ", ".join(sorted(f.replace('_', ' ') for f in matched))
            return "General accessibility fit"

        result["why_recommended"] = result.apply(explain, axis=1)

        result = result.sort_values("final_score", ascending=False).head(top_k)
        cols = ["item_id", "category", "fabric", "color", "style_tag", "price",
                "adaptive_features", "final_score", "why_recommended"]
        return result[cols].reset_index(drop=True)


if __name__ == "__main__":
    rec = AdaptiveFashionRecommender()
    demo = rec.recommend(disability_type="Limited hand dexterity", preferred_style="Casual",
                          budget_tier="Medium", age=45, top_k=5)
    print(demo.to_string(index=False))
