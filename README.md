# California Housing Price Prediction 🏠

This repository contains a Machine Learning project implementing Linear Regression models to predict housing prices using the **California Housing dataset**.

The project focuses on building predictive models from scratch, manual implementation of error metrics, and model optimization using regularization techniques.

## 🚀 Project Overview

The goal of this project is to predict the median house value for California districts based on various features.

### Key Implementations:
* **Data Preprocessing:** Feature selection, handling categorical data, and scaling using `StandardScaler`.
* **Custom Metrics:** Manual implementation of **SSE** (Sum of Squared Errors), **MAE** (Mean Absolute Error), and **MSE** (Mean Squared Error) to understand the math behind the metrics.
* **Univariate Analysis:** Analyzing the predictive power of each feature individually.
* **Multivariate Regression:** Building a comprehensive model using all numerical features.
* **Regularization & Tuning:** Implementing **Ridge (L2)** and **Lasso (L1)** regression with Hyperparameter Tuning using **5-Fold Cross Validation**.

## 📂 Files

* `main.py`: The main script containing the analysis, model training, and visualization logic.
* `housing (1).csv`: The dataset used for training and testing.

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/KULLANICI_ADIN/REPO_ISMI.git](https://github.com/KULLANICI_ADIN/REPO_ISMI.git)
    cd REPO_ISMI
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib scikit-learn
    ```

3.  **Run the analysis:**
    ```bash
    python main.py
    ```

## 📊 Results Summary

* **Best Predictor:** The analysis showed that `median_income` is the strongest single predictor for housing prices.
* **Regularization:** Ridge and Lasso models were tuned to find the optimal alpha parameters, preventing overfitting and providing stable error rates across validation folds.

## 📝 Constraints

This project was developed with specific constraints to demonstrate understanding of core concepts:
* No use of ready-made metric functions (calculations are done manually).
* Specific data splitting ratios.
* Manual implementation of loops for feature analysis.

---
*Machine Learning & Pattern Recognition Project*
