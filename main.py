import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'housing (1).csv')

df = pd.read_csv(file_path)

print("Columns:", df.columns.tolist())

if 'total_bedrooms' in df.columns:
    df.drop('total_bedrooms', axis=1, inplace=True)

data = df.select_dtypes(include=[np.number])

X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

def calculate_errors(y_true, y_pred):
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals**2)
    return sse, mae, mse

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=42
)

results = []
print("\n--- Univariate Regression Results ---")

for col in X.columns:
    x_train_feat = np.array(X_train[col]).reshape(-1, 1)
    x_test_feat = np.array(X_test[col]).reshape(-1, 1)
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_feat)
    x_test_scaled = scaler.transform(x_test_feat)
    
    model = LinearRegression()
    model.fit(x_train_scaled, y_train)
    
    predictions = model.predict(x_test_scaled)
    
    sse, mae, mse = calculate_errors(y_test, predictions)
    
    results.append({
        'feature': col,
        'coef': model.coef_[0],
        'sse': sse,
        'mae': mae,
        'mse': mse
    })
    
    print(f"Feature: {col:15} | Coef: {model.coef_[0]:.2f} | MSE: {mse:.2f}")

df_results = pd.DataFrame(results)

best_sse = df_results.loc[df_results['sse'].idxmin()]
best_mae = df_results.loc[df_results['mae'].idxmin()]
best_mse = df_results.loc[df_results['mse'].idxmin()]

print(f"\nLowest SSE: {best_sse['feature']} ({best_sse['sse']:.2f})")
print(f"Lowest MAE: {best_mae['feature']} ({best_mae['mae']:.2f})")
print(f"Lowest MSE: {best_mse['feature']} ({best_mse['mse']:.2f})")

metrics = ['sse', 'mae', 'mse']
plt.figure(figsize=(15, 5))

for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i+1)
    df_sorted = df_results.sort_values(by=metric)
    plt.bar(df_sorted['feature'], df_sorted[metric])
    plt.title(f'{metric.upper()} by Feature')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(metric.upper())

plt.tight_layout()
plt.show()

scaler_multi = StandardScaler()
X_train_scaled = scaler_multi.fit_transform(X_train)
X_test_scaled = scaler_multi.transform(X_test)

model_multi = LinearRegression()
model_multi.fit(X_train_scaled, y_train)
pred_multi = model_multi.predict(X_test_scaled)

sse_m, mae_m, mse_m = calculate_errors(y_test, pred_multi)

print("\n--- Multivariate Regression Results ---")
print("Coefficients:")
for f, c in zip(X.columns, model_multi.coef_):
    print(f"{f}: {c:.4f}")
    
print(f"\nPerformance -> MSE: {mse_m:.2f}, MAE: {mae_m:.2f}, SSE: {sse_m:.2f}")

scaler_final = StandardScaler()
X_scaled_all = scaler_final.fit_transform(X)

alphas = np.logspace(-3, 5, 50) 

ridge_errors = []
ridge_std = []
lasso_errors = []
lasso_std = []

cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- Calculating Ridge and Lasso Models ---")

for a in alphas:
    model_r = Ridge(alpha=a)
    scores_r = cross_val_score(model_r, X_scaled_all, y, cv=cv, scoring='neg_mean_squared_error')
    ridge_errors.append(-np.mean(scores_r))
    ridge_std.append(np.std(scores_r))
    
    model_l = Lasso(alpha=a)
    scores_l = cross_val_score(model_l, X_scaled_all, y, cv=cv, scoring='neg_mean_squared_error')
    lasso_errors.append(-np.mean(scores_l))
    lasso_std.append(np.std(scores_l))

plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_errors, label='Ridge MSE')
plt.plot(alphas, lasso_errors, label='Lasso MSE')
plt.xscale('log')
plt.title('Regularization: Error vs Alpha')
plt.xlabel('Alpha Parameter')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

best_ridge_idx = np.argmin(ridge_errors)
best_lasso_idx = np.argmin(lasso_errors)

print("\n--- Best Regularization Parameters ---")
print(f"Best Ridge Alpha: {alphas[best_ridge_idx]:.4f}")
print(f"Mean MSE: {ridge_errors[best_ridge_idx]:.2f} (Std: {ridge_std[best_ridge_idx]:.2f})")

print(f"\nBest Lasso Alpha: {alphas[best_lasso_idx]:.4f}")
print(f"Mean MSE: {lasso_errors[best_lasso_idx]:.2f} (Std: {lasso_std[best_lasso_idx]:.2f})")