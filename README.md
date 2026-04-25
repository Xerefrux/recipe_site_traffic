# Recipe Site Traffic Predictor — End-to-End ML Pipeline

A production-style machine learning application that predicts whether a recipe
featured on the **Tasty Bytes** homepage will generate **high site traffic**,
based on its nutritional profile and category. Built with a modular ML
pipeline architecture and deployed as a live web application on Render.

🔗 **Live Demo:** https://recipe-site-traffic.onrender.com/predictdata

---

## What This Project Does

Given a recipe's category, number of servings, and per-serving nutritional
values (calories, carbohydrates, sugar, protein), the app predicts whether
that recipe is likely to drive **high traffic** to the Tasty Bytes homepage —
and displays the model's confidence probability alongside the prediction.

The business context is concrete: Tasty Bytes has observed that featuring a
popular recipe on the homepage increases traffic to the rest of the site by as
much as 40%, which directly translates into new subscriptions. The product
manager's goal is to correctly identify high-traffic recipes **at least 80% of
the time** while minimising the chance of featuring an unpopular one.

This means **Precision** on the High Traffic class is the primary metric
driving every decision in the pipeline — from hyperparameter tuning to model
selection — rather than accuracy (which is misleading under class imbalance)
or R² (which is inappropriate for classification entirely).

---

## Architecture Overview

The project is structured as a modular pipeline with three distinct training
stages that run sequentially, plus a separate inference pipeline that serves
predictions at runtime without retraining anything.

**Stage 1 — Data Ingestion** (`src/components/data_ingestion.py`) reads the
raw dataset and applies all documented cleaning decisions: dropping the recipe
ID column (an identifier, not a feature), merging the undocumented
'Chicken Breast' category into 'Chicken' to align with the data dictionary's
specified 10 categories, extracting the leading integer from servings strings
like '4 as a snack', and binary-encoding the target variable ('High' → 1,
everything else → 0). It then performs a **stratified** 80/20 train-test split
— stratified because the ~60/40 class imbalance means a plain random split
could produce a test set with a misleadingly different class ratio. All three
versions of the data (raw cleaned, train, test) are saved to `artifacts/` for
full reproducibility.

**Stage 2 — Data Transformation** (`src/components/data_transformation.py`)
builds a `scikit-learn` ColumnTransformer and fits it exclusively on the
training split. Numeric features (calories, carbohydrate, sugar, protein,
servings) receive **median imputation** — chosen over mean imputation because
all four nutrition columns are right-skewed, with a small number of very
high-calorie recipes pulling the mean upward — followed by Standard Scaling so
that no feature numerically dominates others purely by magnitude. The
categorical feature (category) receives most-frequent imputation, One-Hot
Encoding with `handle_unknown='ignore'`, and Standard Scaling. The fitted
preprocessor is serialized to `artifacts/preprocessor.pkl` — a critical step
that guarantees inference applies the exact same imputation medians, scaler
parameters, and OHE vocabulary as training, preventing training-serving skew.

**Stage 3 — Model Training** (`src/components/model_trainer.py`) trains six
classifiers in an automated sweep: Logistic Regression (the interpretable
baseline), Random Forest, Gradient Boosting, XGBoost, K-Nearest Neighbors,
and AdaBoost. Hyperparameter tuning is performed on each via `GridSearchCV`
with 5-fold cross-validation, with `scoring="precision"` so the search
directly optimises the business metric. The model with the highest test-set
precision on the High Traffic class is automatically selected and saved as
`artifacts/model.pkl`.

**Inference Pipeline** (`src/pipeline/predict_pipeline.py`) — when a user
submits the web form, the app loads `model.pkl` and `preprocessor.pkl` from
disk, calls `preprocessor.transform()` (never `fit_transform()`, which would
recompute scaling parameters on a single input row and produce wrong feature
values), and returns both a class label and a confidence probability to display
in the UI.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML & Data | scikit-learn, XGBoost, pandas, NumPy |
| Web App | Flask + Gunicorn |
| Deployment | Render (auto-deploys from GitHub) |
| Engineering | Custom exception handler, timestamped file logger, `dill` serialization |

---

## Key Engineering Decisions

**Modular pipeline over a monolithic notebook.** Each training stage is a
self-contained class with its own configuration dataclass. Swapping the model
set, changing the preprocessing strategy, or adding a new feature touches
exactly one file without breaking the rest of the system.

**Precision as the optimisation metric, not accuracy.** A false positive —
predicting High Traffic when the recipe is actually unpopular — means a bad
homepage experience and a missed subscription opportunity. `GridSearchCV` is
configured with `scoring="precision"` so hyperparameter search directly
minimises the error type the business cares about most.

**Stratified train-test split.** With a roughly 60/40 class imbalance in the
target, stratification preserves the class ratio in both splits and ensures
test-set precision reflects real-world conditions rather than a lucky random
partition.

**Strict separation of training and inference.** `PredictPipeline` calls
`.transform()`, never `.fit_transform()`, on the loaded preprocessor. This
guarantees that the imputation medians, scaler parameters, and OHE vocabulary
learned from training data are applied unchanged to every new prediction — a
correctness property that many beginner projects overlook.

**Confidence probabilities alongside class labels.** Rather than returning
only "High" or "Not High", the inference pipeline also surfaces
`predict_proba()` scores. A result like "87% likely to generate High Traffic"
is substantially more actionable for a product manager than a bare label.

**OS-agnostic file paths.** All file paths are constructed with
`os.path.join()` rather than hardcoded slashes, ensuring the project runs
correctly on both Windows (local development) and Linux (Render production).

---

## Results

| Metric | Value |
|---|---|
| Best model | Auto-selected by test-set precision — run training to see winner |
| Business target | ≥ 80% Precision on the High Traffic class |

To see which model won, the full comparison table, and the complete
classification report, run the training pipeline locally — all results are
printed to the console during Stage 3.

---

## Running Locally

```bash
# 1. Clone the repository
git clone <your-github-url>
cd recipe-traffic-predictor

# 2. Install dependencies and register the src package
pip install -r requirements.txt
pip install -e .

# 3. Run the training pipeline — triggers all three stages in sequence
#    and generates artifacts/model.pkl and artifacts/preprocessor.pkl
python src/components/data_ingestion.py

# 4. Start the Flask development server
python application.py
```

Then navigate to `http://localhost:5000/predictdata` and use the prediction
form. Enter any recipe's nutritional values and category to get an instant
traffic prediction with a confidence score.

---

## Project Structure

```
recipe-traffic-predictor/
├── data/
│   └── recipe_site_traffic_2212.csv   # Raw dataset (947 rows × 8 columns)
├── src/
│   ├── components/
│   │   ├── data_ingestion.py          # Stage 1: clean, split, save to artifacts/
│   │   ├── data_transformation.py     # Stage 2: build ColumnTransformer, save preprocessor.pkl
│   │   └── model_trainer.py           # Stage 3: train 6 classifiers, select & save best by precision
│   ├── pipeline/
│   │   ├── predict_pipeline.py        # Inference: load artifacts, transform input, predict
│   │   └── train_pipeline.py          # Reserved for standalone training entry point
│   ├── exception.py                   # Custom exception enriched with file path & line number
│   ├── logger.py                      # Timestamped rotating file logger (one file per run)
│   └── utils.py                       # save_object, load_object, evaluate_models
├── templates/
│   ├── home.html                      # Prediction form with result display (Flask/Jinja2)
│   └── index.html                     # Root redirect to /predictdata
├── artifacts/                         # Auto-generated during training — gitignored
│   ├── model.pkl                      #   Serialized best classifier
│   ├── preprocessor.pkl               #   Serialized fitted ColumnTransformer
│   ├── train.csv                      #   Training split (80%)
│   └── test.csv                       #   Test split (20%)
├── logs/                              # Auto-generated — one log file per training run
├── .ebextensions/
│   └── python.config                  # AWS Elastic Beanstalk WSGI configuration
├── application.py                     # Flask app entry point (WSGI-compatible)
├── setup.py                           # Registers src/ as an installable package
├── Procfile                           # Render/Heroku start command (Gunicorn)
└── requirements.txt                   # Python dependencies
```

---

## Deployment on Render

The project is deployed on **Render** using their free web service tier, with
automatic re-deployment triggered on every push to the `main` branch.

Connect GitHub's repository to Render, set the build command to
`pip install -r requirements.txt && pip install -e .`, and set the start
command to `gunicorn application:application`. The `artifacts/` directory
containing `model.pkl` and `preprocessor.pkl` must be present before the
first request — either commit the pre-trained artifacts to the repository, or
add `python src/components/data_ingestion.py` as a pre-deploy step.

> **Note:** On Render's free tier the app spins down after 15 minutes of
> inactivity and takes approximately 30 seconds to wake on the next request.
> This is expected behaviour for free hosting and does not affect prediction
> correctness.

---

## Dataset

The dataset contains 947 Tasty Bytes recipe records. Features available for
prediction: `calories`, `carbohydrate`, `sugar`, and `protein` (all per
serving), `category` (one of ten recipe types: Chicken, Beverages, Breakfast,
Dessert, Lunch/Snacks, Meat, One Dish Meal, Pork, Potato, Vegetable), and
`servings`. The prediction target is `high_traffic` — a binary label
indicating whether the recipe generated high site traffic when featured on the
homepage.

All data validation and cleaning decisions are documented inline in
`src/components/data_ingestion.py`, including the rationale for each choice,
so the full thought process is auditable alongside the code.