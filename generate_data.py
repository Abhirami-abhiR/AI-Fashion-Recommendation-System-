"""
generate_data.py
-----------------
Generates two synthetic but realistically-structured CSV datasets used to train
and evaluate the Adaptive Fashion Recommender:

  1. users.csv           -> user profiles (disability type, body/mobility info, style preference)
  2. clothing_items.csv  -> a catalog of clothing items tagged with adaptive features
  3. interactions.csv    -> simulated user-item suitability ratings (used as ML training labels)

The domain logic (which adaptive features matter for which disability) is based on
publicly documented adaptive-clothing guidance (e.g. easy closures for limited hand
dexterity, seated-fit cuts for wheelchair users, tagless/seamless fabric for sensory
sensitivity, high-contrast/simple layouts for visual impairment, one-handed/side-entry
designs for limb difference or post-stroke/hemiplegia, etc.).

Run:
    python generate_data.py
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Domain knowledge: disability -> which adaptive features help most
# ---------------------------------------------------------------------------
DISABILITY_TYPES = [
    "Wheelchair user",
    "Limited hand dexterity",
    "Visual impairment",
    "Sensory sensitivity",
    "Limb difference (upper)",
    "Limb difference (lower)",
    "Post-stroke / hemiplegia",
    "Amputee (prosthetic user)",
    "Dwarfism / short stature",
    "None / general accessibility",
]

# Adaptive clothing features present in the catalog
ADAPTIVE_FEATURES = [
    "magnetic_closure",
    "velcro_closure",
    "side_seated_fit",
    "adjustable_hem",
    "tagless",
    "seamless_interior",
    "one_handed_zipper_pull",
    "front_opening",
    "wide_arm_hole",
    "wheelchair_back_length",
    "high_contrast_pattern",
    "simple_texture_id",
    "prosthetic_leg_access_zip",
    "extra_wide_leg_opening",
    "no_buttons",
    "elastic_waist",
    "removable_sleeve",
    "reflective_trim",
]

# Feature relevance weight per disability type (used to simulate ground-truth suitability)
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

CATEGORIES = ["Shirt", "T-Shirt", "Trousers", "Jeans", "Dress", "Jacket", "Skirt", "Hoodie", "Nightwear", "Shoes"]
FABRICS = ["Cotton", "Cotton-Blend", "Fleece", "Denim", "Linen", "Jersey Knit", "Polyester-Blend", "Wool-Blend"]
COLORS = ["Black", "Navy", "Grey", "White", "Red", "Beige", "Olive", "Royal Blue", "Maroon", "Charcoal"]
STYLE_TAGS = ["Casual", "Formal", "Sport", "Loungewear", "Workwear", "Winter", "Summer"]

N_USERS = 400
N_ITEMS = 500


def generate_users(n=N_USERS) -> pd.DataFrame:
    rows = []
    for uid in range(1, n + 1):
        disability = random.choice(DISABILITY_TYPES)
        rows.append({
            "user_id": uid,
            "age": np.random.randint(16, 75),
            "gender": random.choice(["Male", "Female", "Non-binary"]),
            "disability_type": disability,
            "mobility_aid": random.choice(["None", "Wheelchair", "Cane", "Prosthetic", "Walker"])
                if disability in ("Wheelchair user", "Amputee (prosthetic user)") else "None",
            "preferred_style": random.choice(STYLE_TAGS),
            "budget_tier": random.choice(["Low", "Medium", "High"]),
            "size": random.choice(["XS", "S", "M", "L", "XL", "XXL"]),
        })
    return pd.DataFrame(rows)


def generate_items(n=N_ITEMS) -> pd.DataFrame:
    rows = []
    for iid in range(1, n + 1):
        # Each item randomly gets 0-5 adaptive features
        n_feats = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.15, 0.25, 0.25, 0.2, 0.1, 0.05])
        feats = random.sample(ADAPTIVE_FEATURES, k=n_feats)
        rows.append({
            "item_id": iid,
            "category": random.choice(CATEGORIES),
            "fabric": random.choice(FABRICS),
            "color": random.choice(COLORS),
            "style_tag": random.choice(STYLE_TAGS),
            "price": round(np.random.uniform(8, 120), 2),
            "adaptive_features": "|".join(feats) if feats else "none",
        })
    return pd.DataFrame(rows)


def suitability_score(disability: str, item_features: str, style_match: bool, budget_ok: bool) -> float:
    """Simulates a ground-truth suitability rating (1-5) a real occupational
    therapist / user-study panel might assign, used as ML training label."""
    if item_features == "none":
        item_feat_set = set()
    else:
        item_feat_set = set(item_features.split("|"))

    relevant = set(DISABILITY_FEATURE_MAP.get(disability, []))
    if relevant:
        overlap = len(item_feat_set & relevant) / len(relevant)
    else:
        overlap = 0.3  # baseline for general accessibility

    base = 2.0 + 3.0 * overlap
    base += 0.4 if style_match else 0
    base += 0.3 if budget_ok else 0
    noise = np.random.normal(0, 0.35)
    score = np.clip(base + noise, 1, 5)
    return round(score, 2)


def generate_interactions(users: pd.DataFrame, items: pd.DataFrame, samples_per_user=12) -> pd.DataFrame:
    rows = []
    price_bins = {"Low": (0, 30), "Medium": (30, 70), "High": (70, 200)}
    for _, u in users.iterrows():
        sampled_items = items.sample(n=samples_per_user, random_state=u["user_id"])
        for _, it in sampled_items.iterrows():
            style_match = it["style_tag"] == u["preferred_style"]
            lo, hi = price_bins[u["budget_tier"]]
            budget_ok = lo <= it["price"] <= hi
            score = suitability_score(u["disability_type"], it["adaptive_features"], style_match, budget_ok)
            rows.append({
                "user_id": u["user_id"],
                "item_id": it["item_id"],
                "suitability_rating": score,
            })
    return pd.DataFrame(rows)


def main():
    users = generate_users()
    items = generate_items()
    interactions = generate_interactions(users, items)

    users.to_csv(OUT_DIR / "users.csv", index=False)
    items.to_csv(OUT_DIR / "clothing_items.csv", index=False)
    interactions.to_csv(OUT_DIR / "interactions.csv", index=False)

    print(f"Generated {len(users)} users -> users.csv")
    print(f"Generated {len(items)} clothing items -> clothing_items.csv")
    print(f"Generated {len(interactions)} interactions -> interactions.csv")


if __name__ == "__main__":
    main()
