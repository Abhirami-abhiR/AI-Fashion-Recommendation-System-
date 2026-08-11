"""
preprocessing.py
-----------------
Feature engineering utilities shared by training and inference.

Turns raw users.csv / clothing_items.csv rows into numeric feature vectors:
  - Multi-hot encoding of adaptive clothing features
  - One-hot encoding of categorical attributes (category, fabric, style, disability)
  - Simple numeric features (price, age)
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import joblib

ADAPTIVE_FEATURES = [
    "magnetic_closure", "velcro_closure", "side_seated_fit", "adjustable_hem",
    "tagless", "seamless_interior", "one_handed_zipper_pull", "front_opening",
    "wide_arm_hole", "wheelchair_back_length", "high_contrast_pattern",
    "simple_texture_id", "prosthetic_leg_access_zip", "extra_wide_leg_opening",
    "no_buttons", "elastic_waist", "removable_sleeve", "reflective_trim",
]


class FeatureEncoder:
    """Fits/loads encoders for items and users, and builds the combined
    (user, item) feature vector consumed by the ML ranking model."""

    def __init__(self):
        self.mlb = MultiLabelBinarizer(classes=ADAPTIVE_FEATURES)
        self.item_cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.user_cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self._fitted = False

    # ------------------------------------------------------------------ #
    def fit(self, users: pd.DataFrame, items: pd.DataFrame):
        feat_lists = items["adaptive_features"].apply(
            lambda s: [] if s == "none" else s.split("|")
        )
        self.mlb.fit(feat_lists)

        self.item_cat_encoder.fit(items[["category", "fabric", "style_tag"]])
        self.user_cat_encoder.fit(users[["disability_type", "preferred_style", "budget_tier"]])
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    def transform_items(self, items: pd.DataFrame) -> pd.DataFrame:
        feat_lists = items["adaptive_features"].apply(
            lambda s: [] if s == "none" else s.split("|")
        )
        multi_hot = self.mlb.transform(feat_lists)
        multi_hot_df = pd.DataFrame(multi_hot, columns=self.mlb.classes_, index=items.index)

        cat_encoded = self.item_cat_encoder.transform(items[["category", "fabric", "style_tag"]])
        cat_cols = self.item_cat_encoder.get_feature_names_out(["category", "fabric", "style_tag"])
        cat_df = pd.DataFrame(cat_encoded, columns=cat_cols, index=items.index)

        numeric = items[["price"]].reset_index(drop=True)
        numeric.index = items.index

        out = pd.concat([items[["item_id"]], multi_hot_df, cat_df, numeric], axis=1)
        return out

    # ------------------------------------------------------------------ #
    def transform_users(self, users: pd.DataFrame) -> pd.DataFrame:
        cat_encoded = self.user_cat_encoder.transform(
            users[["disability_type", "preferred_style", "budget_tier"]]
        )
        cat_cols = self.user_cat_encoder.get_feature_names_out(
            ["disability_type", "preferred_style", "budget_tier"]
        )
        cat_df = pd.DataFrame(cat_encoded, columns=cat_cols, index=users.index)
        numeric = users[["age"]].reset_index(drop=True)
        numeric.index = users.index
        out = pd.concat([users[["user_id"]], cat_df, numeric], axis=1)
        return out

    # ------------------------------------------------------------------ #
    def build_pair_features(self, user_row: pd.DataFrame, items_encoded: pd.DataFrame) -> pd.DataFrame:
        """Cross-joins a single encoded user row with every encoded item row
        to build the (user, item) feature matrix expected by the ML model."""
        user_repeated = pd.concat([user_row] * len(items_encoded), ignore_index=True)
        user_repeated.index = items_encoded.index
        combined = pd.concat(
            [items_encoded.drop(columns=["item_id"]), user_repeated.drop(columns=["user_id"])],
            axis=1,
        )
        return combined

    # ------------------------------------------------------------------ #
    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FeatureEncoder":
        return joblib.load(path)
