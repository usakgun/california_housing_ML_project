# California Housing Price Prediction 🏠

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Lib-Scikit--Learn%20|%20Pandas%20|%20Matplotlib-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

This project implements an end-to-end Machine Learning pipeline to predict median house values in Californian districts. Using data from the 1990 California census, the model learns from various features such as population, median income, and geographical location.

## 📌 Project Overview

The goal is to build a regression model that can predict the `median_house_value` for a given district. This serves as a fundamental exercise in structuring a complete ML project, covering everything from data fetching to final model evaluation.

### Key Steps Implemented
1.  **Exploratory Data Analysis (EDA):** Visualizing geographical data (latitude/longitude), correlations, and attribute distributions.
2.  **Data Preprocessing:**
    * Handling missing values using simple imputation.
    * Handling categorical data (`ocean_proximity`) via OneHotEncoding.
    * Feature Scaling using Standardization.
3.  **Feature Engineering:** Creating new informative features like `rooms_per_household`, `bedrooms_per_room`, and `population_per_household`.
4.  **Transformation Pipelines:** Automating the data flow using Scikit-Learn's `Pipeline` and `ColumnTransformer`.
5.  **Model Training & Evaluation:**
    * Linear Regression
    * Decision Tree Regressor
    * **Random Forest Regressor** (Best Performing)

## 📂 Dataset

The dataset is based on the 1990 California Census data. It contains 20,640 entries with 10 features, including:
* `longitude` / `latitude`: Geolocation.
* `housing_median_age`: Median age of a house within a block.
* `total_rooms` / `total_bedrooms`: Total number of rooms/bedrooms.
* `population` / `households`: Demographics.
* `median_income`: Median income for households (Key predictor).
* `ocean_proximity`: Location relative to the ocean.

## 🚀 Methodology & Results

The models were evaluated using **Root Mean Squared Error (RMSE)**.

| Model | Performance | Observation |
| :--- | :--- | :--- |
| **Linear Regression** | High RMSE (Underfitting) | The data is not strictly linear. |
| **Decision Tree** | Low Training RMSE / High Val RMSE | Severe overfitting to the training data. |
| **Random Forest** | **Best RMSE** | Generalized well by averaging multiple trees. |

> **Final Optimization:** The Random Forest model was further tuned using `GridSearchCV` to find the optimal hyperparameters (e.g., `n_estimators`, `max_features`).

## 🛠️ Tech Stack

* **Python:** Core programming language.
* **Pandas & NumPy:** Data manipulation and numerical operations.
* **Matplotlib & Seaborn:** Data visualization (histograms, scatter plots, heatmaps).
* **Scikit-Learn:** Machine Learning library for pipelines, models, and metrics.

## ⚙️ Installation

1.  Clone the repo:
    ```bash
    git clone [https://github.com/usakgun/california_housing_ML_project.git](https://github.com/usakgun/california_housing_ML_project.git)
    ```
2.  Install requirements:
    ```bash
    pip install pandas numpy scikit-learn matplotlib jupyter
    ```
3.  Run the notebook:
    ```bash
    jupyter notebook
    ```
