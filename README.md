# 🏠 House Price Prediction

Predicting residential property sale prices in Ames, Iowa, using the classic [Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). This project implements a complete machine learning workflow — from exploratory data analysis through to model evaluation and interpretation.

---

## 📋 Overview

The goal is to predict the sale price of a house (`SalePrice`) from 79 explanatory variables describing its physical characteristics, location, and condition. The project demonstrates the full supervised learning pipeline, with a strong focus on **context-aware data cleaning** and **principled model selection**.

---

## 📊 Dataset

- **Source:** Kaggle — House Prices: Advanced Regression Techniques
- **Training set:** 1,460 properties · 79 features
- **Target variable:** `SalePrice` (USD)

---

## 🔧 Workflow

1. **Imports & Data Loading**
2. **Exploratory Data Analysis** — target distribution, correlations, feature relationships
3. **Data Cleaning & Missing Values** — context-aware imputation
4. **Feature Engineering** — ordinal mapping + one-hot encoding
5. **Modelling & Evaluation** — baseline, complex model, hyperparameter tuning
6. **Conclusions**

---

## 🧠 Key Decisions

- **Log transformation of the target.** `SalePrice` was right-skewed; applying `log1p` produced a near-normal distribution that improved model performance.

- **Context-aware missing value handling.** Many missing values were not truly missing — they encoded the *absence* of a feature (e.g. no pool, no garage). These were filled with `"None"` rather than dropped or imputed, preserving meaningful information.

- **Encoding by feature type.** Ordinal features (quality ratings) were mapped manually to preserve their natural order; nominal features were one-hot encoded to avoid imposing false ordinal relationships.

---

## 📈 Results

Two models were trained and evaluated on a held-out validation set. Errors (MAE, RMSE) are expressed in USD after reverting the log transformation.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** | **~$16,283** | **~$24,840** | **0.9196** |
| Random Forest | ~$17,244 | ~$29,378 | 0.8875 |
| Random Forest (Tuned) | ~$16,765 | ~$30,200 | 0.8811 |

**The simple Linear Regression outperformed the more complex Random Forest** across RMSE and R², even after hyperparameter tuning via grid search. This reinforces the principle of **parsimony**: added complexity is only justified when it produces a measurable improvement. The log transformation had already linearised many of the feature–target relationships, playing to the linear model's strengths.

---

## 🛠️ Technologies

- **Python**
- **pandas** / **NumPy** — data manipulation
- **Matplotlib** / **seaborn** — visualisation
- **scikit-learn** — modelling, evaluation, hyperparameter tuning

---

## 📁 Project Structure

```
house-price-prediction/
│
├── data/                  # Kaggle CSVs (not tracked)
├── notebooks/
│   └── analysis.ipynb     # Main analysis notebook
├── images/                # Exported figures
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd house-price-prediction
   ```

2. Install the dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) and place `train.csv` and `test.csv` in the `data/` folder.

4. Launch the notebook:
   ```bash
   jupyter notebook notebooks/analysis.ipynb
   ```

---

## 🔮 Possible Improvements

- Apply feature scaling and analyse linear regression coefficients (or permutation importance) to compare feature drivers across both models.
- Treat specific numerical features more precisely (e.g. filling garage-related variables with `0` rather than the median when no garage exists).
- Experiment with gradient boosting models (XGBoost, LightGBM), which often excel on tabular data.
- Address remaining outliers identified during EDA.