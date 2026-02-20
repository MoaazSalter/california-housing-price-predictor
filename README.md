# California Housing Price Predictor

A machine learning regression project that predicts median house values across California districts using the classic California Housing dataset. The project covers the full ML pipeline — from data cleaning and feature engineering through multi-model comparison, hyperparameter tuning, and final model export.

---

## Results

| Split | RMSE |
|---|---|
| Training | ~45,022 |
| Cross-Validation | ~48,790 |
| **Test (final)** | **~50,497** |

> The typical baseline RMSE for this dataset is ~115,000. This model beats that benchmark.

---

## Project Structure

```
california-housing-price-predictor/
│
├── California_housing_evaluator.ipynb # Main notebook
│
├── data/
│   ├──housing.csv                     # Raw dataset
│
├── models/
│   ├── Housing_estimator_model.pkl    # Saved best model
│   └── model_info.pkl                 # Best params + scores
│
└── plots/
    ├── all_models_ranking.png         # All candidates ranked by RMSE
    ├── top6_models.png                # Train vs validation for top 6
    └── models_distribution.png        # Score distribution by model type
```

---

## Pipeline Overview

### 1 — Loading the Dataset
The dataset is loaded from `housing.csv` and inspected via `.head()`, `.describe()`, and `.info()`.

### 2 — Data Imputation
Missing values in `total_bedrooms` are filled using the **mode of records sharing the same `total_rooms` value**, preserving the relationship between the two features rather than using a global mean/median. Remaining nulls (records with no matching group) are dropped.

### 3 — EDA & Outlier Handling
Two data quality issues were identified and resolved:
- `median_house_value` was artificially capped at $500,000 — rows at the cap were removed.
- `housing_median_age` was artificially capped at 52 — rows at the cap were removed.
- `ISLAND` category in `ocean_proximity` had only 2 observations and was dropped to prevent overfitting.

### 4 — Feature Engineering
Two new features were created:
- **`distance_to_center`** — Euclidean distance from each district to the geographic center of the dataset.
- **`bedroom_per_room`** — ratio of bedrooms to total rooms, a more meaningful signal than raw counts.

`ocean_proximity` was one-hot encoded using `pd.get_dummies`.

### 5 & 6 — Modelling
A `sklearn` `Pipeline` combining `StandardScaler` + a model placeholder was used alongside `GridSearchCV` with **6-fold KFold cross-validation** to evaluate 41 candidate configurations across 5 model families:

| Model | Hyperparameters Searched |
|---|---|
| LinearRegression | — |
| Lasso | alpha: 0.001, 0.01, 0.1, 1 |
| Ridge | alpha: 0.001, 0.01, 0.1, 1 |
| SVR | kernel: linear/rbf, C: 10–10000, epsilon: 0.001–0.1 |
| RandomForestRegressor | max_depth, min_samples_split, min_samples_leaf |

Scoring metric: `neg_root_mean_squared_error`

> **Note on SVR:** SVR requires large `C` values (10–10,000) because the target variable is in the hundreds of thousands. Small `C` values cause severe underfitting.

### 7 — Results & Visualisation
Three plots are generated:
- All 41 model configurations ranked by validation RMSE
- Top 6 models comparing training vs validation RMSE side-by-side
- Boxplot of score distribution per model family across all CV folds

### 8 — Model Export
The best estimator and its metadata (best params + all three RMSE scores) are saved to the `models/` folder using `joblib`.

---

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
joblib
```

Install with:
```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

---

## How to Run

1. Clone the repo
```bash
git clone https://github.com/MomoSalter/california-housing-price-predictor.git
cd california-housing-price-predictor
```

2. Install dependencies
```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

3. Open the notebook
```bash
jupyter notebook California_housing_evaluator.ipynb
```

4. Run all cells top to bottom.

---

## Dataset

The dataset used is the **California Housing dataset**, originally from the 1990 U.S. Census. Each row represents a census block district and includes:

| Feature | Description |
|---|---|
| `longitude` / `latitude` | Geographic location |
| `housing_median_age` | Median age of houses in the district |
| `total_rooms` | Total rooms across all households |
| `total_bedrooms` | Total bedrooms across all households |
| `population` | District population |
| `households` | Number of households |
| `median_income` | Median income (in tens of thousands USD) |
| `ocean_proximity` | Categorical proximity to the ocean |
| `median_house_value` | **Target** — median house value in USD |

## Author

**Moaaz Ahmed**
- GitHub: [@Moaaz Ahmed](https://github.com/MomoSalter)
