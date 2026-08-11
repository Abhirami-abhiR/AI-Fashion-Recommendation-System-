"""
test_recommender.py
--------------------
Minimal sanity tests. Run with:  pytest tests/
(Assumes `python data/generate_data.py` and `python src/train_model.py`
have already been run so models/ and data/ are populated.)
"""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "src"))

from recommender import AdaptiveFashionRecommender  # noqa: E402

MODEL_DIR = str(ROOT / "models")
DATA_DIR = str(ROOT / "data")


def test_recommender_loads():
    rec = AdaptiveFashionRecommender(model_dir=MODEL_DIR, data_dir=DATA_DIR)
    assert rec.items is not None and len(rec.items) > 0


def test_recommend_returns_topk():
    rec = AdaptiveFashionRecommender(model_dir=MODEL_DIR, data_dir=DATA_DIR)
    results = rec.recommend(disability_type="Wheelchair user", top_k=5)
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 5
    assert "final_score" in results.columns
    assert "why_recommended" in results.columns


def test_scores_are_sorted_descending():
    rec = AdaptiveFashionRecommender(model_dir=MODEL_DIR, data_dir=DATA_DIR)
    results = rec.recommend(disability_type="Visual impairment", top_k=10)
    scores = results["final_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_category_filter_applied():
    rec = AdaptiveFashionRecommender(model_dir=MODEL_DIR, data_dir=DATA_DIR)
    results = rec.recommend(disability_type="Limited hand dexterity", category="Shoes", top_k=5)
    assert (results["category"] == "Shoes").all()


def test_relevant_features_score_higher_than_random_profile():
    rec = AdaptiveFashionRecommender(model_dir=MODEL_DIR, data_dir=DATA_DIR)
    targeted = rec.recommend(disability_type="Sensory sensitivity", top_k=5)
    generic = rec.recommend(disability_type="None / general accessibility", top_k=5)
    assert targeted["final_score"].mean() >= generic["final_score"].mean()
