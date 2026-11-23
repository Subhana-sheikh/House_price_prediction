# House Price Prediction

A machine learning project that explores housing market data and predicts residential property prices using **Linear Regression** and **Stochastic Gradient Descent (SGD) Regression** with a `scikit-learn` preprocessing pipeline.

**Notebook:** [House\_price\_prediction.ipynb](https://www.google.com/search?q=.%2FHouse_price_prediction.ipynb)

## Project Overview

This notebook walks through an end-to-end supervised regression workflow on `Housing.csv`:

1. **Exploratory Data Analysis (EDA):** Inspects dataset shape, data types, summary statistics, missing values, and duplicates, followed by distribution and relationship plots.
2. **Preprocessing Pipeline:** Splits data into training and test sets, scales numerical features with `StandardScaler`, and encodes categorical features with `OneHotEncoder` inside a `ColumnTransformer`.
3. **Model Training & Evaluation:** Trains a baseline `LinearRegression` model, evaluates it with MAE, MSE, RMSE, and $R^2$ score, and inspects learned feature coefficients.
4. **Gradient Descent Modeling:** Trains an `SGDRegressor` with target scaling (`TransformedTargetRegressor`) and rescales coefficients back to the original price scale.
5. **Inference:** Predicts prices for sample house configurations using both trained pipelines.

## Dataset (`Housing.csv`)

* **Size:** 545 rows × 13 columns
* **Data Quality:** 0 missing values, 0 duplicate rows
* **Target Variable:** `price` (range: `1,750,000` – `13,300,000`; mean: `4,766,729`; median: `4,340,000`)

### Feature Summary

| Feature | Type | Values / Range | Description |
| :--- | :--- | :--- | :--- |
| `area` | Numerical (`int64`) | `1,650` – `16,200` | Total property area |
| `bedrooms` | Numerical (`int64`) | `1` – `6` | Number of bedrooms |
| `bathrooms` | Numerical (`int64`) | `1` – `4` | Number of bathrooms |
| `stories` | Numerical (`int64`) | `1` – `4` | Number of floors/stories |
| `parking` | Numerical (`int64`) | `0` – `3` | Number of parking spaces |
| `mainroad` | Categorical (`object`) | `yes`, `no` | Connected to a main road |
| `guestroom` | Categorical (`object`) | `yes`, `no` | Includes a guest room |
| `basement` | Categorical (`object`) | `yes`, `no` | Includes a basement |
| `hotwaterheating` | Categorical (`object`) | `yes`, `no` | Has hot water heating |
| `airconditioning` | Categorical (`object`) | `yes`, `no` | Has air conditioning |
| `prefarea` | Categorical (`object`) | `yes`, `no` | Located in a preferred area |
| `furnishingstatus` | Categorical (`object`) | `furnished`, `semi-furnished`, `unfurnished` | Furnishing condition |

## Exploratory Data Analysis (EDA)

The notebook generates the following visualizations using `matplotlib` and `seaborn`:

* **Price Distribution:** Histogram with KDE (`sns.histplot`) and boxplot (`sns.boxplot`) to inspect skewness and outliers.
* **Area vs. Price:** Scatter plot showing the positive relationship between property area and price.
* **Bedrooms vs. Price:** Scatter plot and boxplot comparing price distributions across bedroom counts (1–6).
* **Correlation Heatmap:** Annotated correlation matrix (`sns.heatmap`) across all numerical features.

## Preprocessing & Modeling Pipeline

* **Train / Test Split:** `80%` training (`436` samples) / `20%` testing (`109` samples) with `random_state=42`.
* **Feature Engineering (`ColumnTransformer`):**
  * **Numerical (`area`, `bedrooms`, `bathrooms`, `stories`, `parking`):** `StandardScaler()`
  * **Categorical (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`):** `OneHotEncoder(handle_unknown="ignore", drop="first")`

### Models Implemented

1. **Linear Regression (`LinearRegression`)**
   * Wrapped in a `Pipeline` with the `ColumnTransformer` preprocessor.
2. **SGD Regressor (`SGDRegressor` + `TransformedTargetRegressor`)**
   * Hyperparameters: `loss="squared_error"`, `penalty=None`, `max_iter=5000`, `learning_rate="constant"`, `eta0=0.01`, `tol=1e-6`, `random_state=42`.
   * Target `price` is standardized during training via `TransformedTargetRegressor(transformer=StandardScaler())`.

## Results & Evaluation

### Linear Regression Test Metrics

| Metric | Value |
| :--- | :--- |
| **MAE** (Mean Absolute Error) | `970,043.40` |
| **MSE** (Mean Squared Error) | `1,754,318,687,330.67` |
| **RMSE** (Root Mean Squared Error) | `1,324,506.96` |
| $R^2$ **Score** | `0.6529` |

### Key Feature Insights (Linear Regression)

* **Intercept ($b_0$):** `3,969,403.56`
* **Strongest Positive Drivers:** `airconditioning_yes` (`+791,426.74`), `hotwaterheating_yes` (`+684,649.89`), `prefarea_yes` (`+629,890.57`), `bathrooms` (`+521,879.03`), and `area` (`+519,552.42`).
* **Negative Drivers:** `furnishingstatus_unfurnished` (`-413,645.06`) and `furnishingstatus_semi-furnished` (`-126,881.82`) relative to furnished homes.

### Sample Predictions

* **Linear Regression** (6,500 sq ft, 3 bedrooms, 2 bathrooms, 2 stories, main road, hot water heating, AC, 2 parking, preferred area, furnished): **`7,951,579.41`**
* **SGD Regressor** (5,000 sq ft, 3 bedrooms, 2 bathrooms, 2 stories, main road, AC, 1 parking, preferred area, furnished): **`6,560,499.11`**

## Tech Stack

* **Language:** Python 3
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`

## How to Run

1. Install the required packages:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

2. Place `Housing.csv` in the same directory as [House\_price\_prediction.ipynb](https://www.google.com/search?q=.%2FHouse_price_prediction.ipynb).

3. Open `House_price_prediction.ipynb` in VS Code (with the Jupyter extension) or launch Jupyter Notebook:

   ```bash
   jupyter notebook House_price_prediction.ipynb
   ```