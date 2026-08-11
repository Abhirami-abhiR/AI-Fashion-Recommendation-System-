# 🧥 AI Fashion Recommendation System for Persons with Disabilities

A **hybrid Machine Learning recommender system** that suggests adaptive clothing
to users based on their disability type, mobility needs, style preference, and
budget — with a plain-language explanation for every recommendation.

Built as a final-year project to demonstrate applied ML (content-based
filtering + a trained ranking model) on a socially meaningful, under-served
problem: mainstream fashion recommenders ignore accessibility needs such as
one-handed closures, seated-fit cuts for wheelchair users, tagless/seamless
fabric for sensory sensitivity, or high-contrast patterns for visual
impairment.

---

## ✨ Key Features

- **Disability-aware needs modeling** — 10 disability/accessibility profiles
  (wheelchair user, limited hand dexterity, visual impairment, sensory
  sensitivity, limb difference, post-stroke/hemiplegia, prosthetic user,
  short stature, etc.) mapped to 18 concrete adaptive clothing features
  (magnetic closures, wheelchair-back length, front opening, tagless,
  high-contrast pattern, etc.)
- **Hybrid recommendation engine**
  - *Content-based filtering*: cosine similarity between a user's needs
    vector and each item's adaptive-feature vector — explainable and works
    instantly for brand-new users (no cold-start problem).
  - *Learned ranking model*: a `RandomForestRegressor` trained on simulated
    interaction data, capturing patterns like style/budget fit and feature
    interactions that pure similarity misses.
  - Final score = weighted blend of both signals.
- **Explainable output** — every recommendation states *why* it was suggested
  ("Matches your needs: magnetic closure, elastic waist").
- **Interactive Streamlit demo app** for live presentation/viva.
- **Offline evaluation** (MAE, R², Precision@K) and unit tests.
- **CI pipeline** (GitHub Actions) that regenerates data, retrains the model,
  and runs tests on every push.

---

## 🏗️ Architecture

```
                ┌─────────────────────┐
                │   User Profile       │
                │ (disability, style,  │
                │  budget, age)        │
                └──────────┬───────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                                      ▼
┌──────────────────┐                 ┌─────────────────────┐
│ Content-Based      │                 │ ML Ranking Model     │
│ Filter              │                 │ (RandomForest)        │
│ (cosine similarity  │                 │ trained on simulated  │
│ needs vs. item      │                 │ (user,item)->rating   │
│ adaptive features)   │                 │ interactions           │
└─────────┬──────────┘                 └───────────┬───────────┘
          │                                          │
          └───────────────────┬──────────────────────┘
                               ▼
                  final_score = α·content + (1-α)·ML
                               │
                               ▼
                  Ranked, explained recommendations
```

---

## 📁 Project Structure

```
fashion-recommender-disabilities/
├── data/
│   ├── generate_data.py      # synthetic dataset generator (users, items, interactions)
│   ├── users.csv              # generated
│   ├── clothing_items.csv     # generated
│   └── interactions.csv       # generated
├── src/
│   ├── preprocessing.py      # feature encoding (FeatureEncoder)
│   ├── train_model.py        # trains & saves the RandomForest ranking model
│   ├── recommender.py        # AdaptiveFashionRecommender hybrid engine
│   └── evaluate.py           # offline metrics: MAE, R², Precision@K
├── app/
│   └── app.py                 # Streamlit demo UI
├── models/                    # saved model.pkl / encoder.pkl (generated)
├── tests/
│   └── test_recommender.py   # pytest unit tests
├── .github/workflows/ci.yml  # CI: regenerate data, train, test
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone & install
```bash
git clone https://github.com/<your-username>/fashion-recommender-disabilities.git
cd fashion-recommender-disabilities
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python data/generate_data.py
```
This creates `users.csv`, `clothing_items.csv`, and `interactions.csv` — a
synthetic but domain-informed dataset (400 users, 500 clothing items, ~4,800
interactions) since no public adaptive-fashion dataset currently exists.
The disability → feature relevance mapping is based on documented adaptive
clothing guidance and is easy to extend or replace with real occupational
therapist / survey data (see [Extending with Real Data](#-extending-with-real-data)).

### 3. Train the model
```bash
python src/train_model.py
```
Trains a `RandomForestRegressor` on the (user, item) → suitability-rating
interactions and saves the model + encoders to `models/`.

### 4. Run the demo app
```bash
streamlit run app/app.py
```
Opens an interactive UI where you pick a disability profile, style, and
budget, and get ranked, explained clothing recommendations.

### 5. Run tests & evaluation
```bash
pytest tests/ -v
python src/evaluate.py
```

---

## 🧠 How the Recommendation Logic Works

1. Each **disability type** is mapped to the adaptive features most relevant
   to it, e.g.:

   | Disability | Prioritized features |
   |---|---|
   | Wheelchair user | seated fit, wheelchair-back length, adjustable hem, elastic waist, front opening |
   | Limited hand dexterity | magnetic closure, velcro closure, no buttons, one-handed zipper pull |
   | Visual impairment | high-contrast pattern, simple texture ID, tagless |
   | Sensory sensitivity | tagless, seamless interior, simple texture ID |
   | Amputee (prosthetic user) | prosthetic leg access zip, extra-wide leg opening, adjustable hem |

2. **Content-based score**: cosine similarity between the user's needs vector
   and each item's multi-hot adaptive-feature vector.
3. **ML score**: the trained model predicts a 1–5 suitability rating for the
   (user, item) pair using all encoded features (adaptive features, category,
   fabric, style, price, disability type, age, budget).
4. **Final score**: `0.5 × content_score + 0.5 × ml_score` (weight is
   configurable via the `alpha` parameter in `recommender.py`).

---

## 📊 Model Performance (on synthetic data)

| Metric | Value |
|---|---|
| MAE (rating scale 1–5) | ~0.36 |
| R² | ~0.53 |

Run `python src/train_model.py` to reproduce — exact numbers vary slightly by
random seed and by any changes to the dataset.

---

## 🔌 Extending with Real Data

This project ships with a **synthetic** dataset because no public
adaptive-fashion interaction dataset exists. To move toward production /
research quality:

- Replace `data/generate_data.py`'s output with real survey/user-study
  ratings (Google Forms, occupational-therapist panels, adaptive-clothing
  retailer reviews).
- Swap in a real product catalog (scrape or partner with adaptive-clothing
  brands, e.g. Tommy Hilfiger Adaptive, Zappos Adaptive) and tag items with
  the same `adaptive_features` vocabulary used here.
- Consider adding a collaborative-filtering component (matrix factorization)
  once real user-item interaction history is available at scale.
- Add user feedback loop: log which recommendations users actually
  click/purchase and periodically retrain the model on that signal.

---

## 🛣️ Future Work

- Body-measurement-based sizing recommendations.
- Image-based garment tagging (CNN) to auto-extract adaptive features from
  product photos.
- Multi-language / screen-reader-optimized UI for visual impairment users.
- Deploy as a public web app (Streamlit Community Cloud / Hugging Face
  Spaces) with a real catalog.

---


