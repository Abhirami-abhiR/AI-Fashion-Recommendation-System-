"""
app.py
------
Streamlit demo UI for the Adaptive Fashion Recommender.

Run from the project root:
    streamlit run app/app.py
"""

from pathlib import Path
import sys
import streamlit as st
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "src"))
from recommender import AdaptiveFashionRecommender, DISABILITY_FEATURE_MAP

MODEL_DIR = str(Path(__file__).parent.parent / "models")
DATA_DIR = str(Path(__file__).parent.parent / "data")

st.set_page_config(page_title="Adaptive Fashion Recommender", page_icon="🧥", layout="wide")

st.title("🧥 AI Fashion Recommendation System for People with Disabilities")
st.caption(
    "A hybrid Machine Learning recommender (content-based filtering + learned ranking model) "
    "that suggests adaptive clothing based on a person's specific needs."
)


@st.cache_resource
def load_recommender():
    return AdaptiveFashionRecommender(model_dir=MODEL_DIR, data_dir=DATA_DIR)


rec = load_recommender()

with st.sidebar:
    st.header("Your Profile")
    disability = st.selectbox("Disability / accessibility need", list(DISABILITY_FEATURE_MAP.keys()))
    style = st.selectbox("Preferred style", ["Casual", "Formal", "Sport", "Loungewear", "Workwear", "Winter", "Summer"])
    budget = st.selectbox("Budget", ["Low", "Medium", "High"])
    age = st.slider("Age", 16, 80, 30)
    category = st.selectbox("Category (optional filter)",
                             ["All", "Shirt", "T-Shirt", "Trousers", "Jeans", "Dress",
                              "Jacket", "Skirt", "Hoodie", "Nightwear", "Shoes"])
    top_k = st.slider("Number of recommendations", 3, 20, 8)
    go = st.button("Get Recommendations", type="primary", use_container_width=True)

if go:
    cat = None if category == "All" else category
    with st.spinner("Scoring items..."):
        results = rec.recommend(disability_type=disability, preferred_style=style,
                                 budget_tier=budget, age=age, category=cat, top_k=top_k)

    st.subheader(f"Top {len(results)} recommendations for: {disability}")

    relevant = DISABILITY_FEATURE_MAP.get(disability, [])
    if relevant:
        st.info("Key features prioritized for this profile: " + ", ".join(f.replace("_", " ") for f in relevant))

    for _, row in results.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{row['category']} — {row['fabric']}, {row['color']}**")
                st.caption(f"Style: {row['style_tag']}  |  Price: ${row['price']:.2f}")
                st.markdown(f"✅ {row['why_recommended']}")
            with c2:
                st.metric("Match score", f"{row['final_score']*100:.0f}%")
else:
    st.markdown(
        """
        👈 Set your profile in the sidebar and click **Get Recommendations**.

        **How it works**
        1. Your disability type is mapped to a set of relevant *adaptive clothing features*
           (e.g. magnetic closures, seated fit, tagless fabric, high-contrast patterns).
        2. A **content-based filter** measures cosine similarity between your needs and each
           item's features.
        3. A **trained Random Forest model** predicts a suitability rating learned from
           historical interaction data (style match, price fit, feature interactions).
        4. The two scores are blended into a final ranked list with a plain-language
           explanation for every recommendation.
        """
    )

st.divider()
st.caption("Final Year Project — AI Fashion Recommendation System for Persons with Disabilities")
