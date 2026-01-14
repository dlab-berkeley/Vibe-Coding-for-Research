import numpy as np
import pandas as pd
from livedemo.linear_regression import LinearRegression as CustomLR
from sklearn.linear_model import LinearRegression as SklearnLR

def compare_implementations():
    print("--- Comparing Custom LinearRegression vs Scikit-Learn ---")
    
    # 1. Generate Test Data
    np.random.seed(42) # Ensure reproducibility
    n_samples = 1000
    X = np.random.randn(n_samples, 1)
    true_slope = 2.0
    true_intercept = 5.0
    noise = np.random.normal(0, 1.0, n_samples)
    y = true_slope * X.flatten() + true_intercept + noise
    
    # 2. Fit Custom Model
    custom_model = CustomLR()
    custom_model.fit(X, y)
    
    # 3. Fit Sklearn Model
    sklearn_model = SklearnLR()
    sklearn_model.fit(X, y)
    
    # 4. Compare Parameters
    # Coefficients
    custom_coef = custom_model.coefficients[0]
    sklearn_coef = sklearn_model.coef_[0]
    coef_diff = abs(custom_coef - sklearn_coef)
    
    # Intercept
    custom_intercept = custom_model.intercept
    sklearn_intercept = sklearn_model.intercept_
    intercept_diff = abs(custom_intercept - sklearn_intercept)
    
    # Predictions (on first 5 points)
    X_test = X[:5]
    custom_pred = custom_model.predict(X_test)
    sklearn_pred = sklearn_model.predict(X_test)
    pred_diff = np.max(np.abs(custom_pred - sklearn_pred))
    
    # 5. Report Results
    tolerance = 1e-10
    
    print(f"{ 'Metric':<15} {'Custom':<15} {'Sklearn':<15} {'Diff':<15} {'Match?':<10}")
    print("---------------------------------------------------------------------------")
    
    print(f"{ 'Slope':<15} {custom_coef:<15.10f} {sklearn_coef:<15.10f} {coef_diff:<15.10e} {str(coef_diff < tolerance):<10}")
    print(f"{ 'Intercept':<15} {custom_intercept:<15.10f} {sklearn_intercept:<15.10f} {intercept_diff:<15.10e} {str(intercept_diff < tolerance):<10}")
    print(f"{ 'Max Pred Diff':<15} {'-':<15} {'-':<15} {pred_diff:<15.10e} {str(pred_diff < tolerance):<10}")
    
    if coef_diff < tolerance and intercept_diff < tolerance and pred_diff < tolerance:
        print("\nSUCCESS: Custom implementation matches Scikit-Learn to numerical precision!")
    else:
        print("\nFAILURE: Significant differences found.")

if __name__ == "__main__":
    compare_implementations()
