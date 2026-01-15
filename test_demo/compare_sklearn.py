import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from linear_regression import LinearRegression as OurLinearRegression

def compare_implementations():
    # 1. Generate test data
    np.random.seed(42)
    n = 1000
    slope = 2.0
    intercept = 5.0
    X = np.random.normal(0, 1, (n, 1))
    y = slope * X.flatten() + intercept + np.random.normal(0, 1, n)

    # 2. Fit both models
    # Our model
    our_model = OurLinearRegression()
    our_model.fit(X, y)
    
    # Sklearn model
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X, y)

    # 3. Compare values
    our_coef = our_model.coefficients[0]
    our_intercept = our_model.intercept
    
    sklearn_coef = sklearn_model.coef_[0]
    sklearn_intercept = sklearn_model.intercept_

    # Compare predictions
    X_test = np.array([[1.5], [2.5], [-0.5]])
    our_pred = our_model.predict(X_test)
    sklearn_pred = sklearn_model.predict(X_test)

    # 4. Report
    tolerance = 1e-10
    
    coef_match = np.isclose(our_coef, sklearn_coef, atol=tolerance)
    intercept_match = np.isclose(our_intercept, sklearn_intercept, atol=tolerance)
    pred_match = np.allclose(our_pred, sklearn_pred, atol=tolerance)

    print("=" * 60)
    print("COMPARISON: OUR MODEL VS SKLEARN")
    print("=" * 60)
    print(f"{ 'Metric':<15} | { 'Our Model':<12} | { 'Sklearn':<12} | {'Match?'}")
    print("-" * 60)
    print(f"{ 'Slope':<15} | {our_coef:<12.8f} | {sklearn_coef:<12.8f} | {coef_match}")
    print(f"{ 'Intercept':<15} | {our_intercept:<12.8f} | {sklearn_intercept:<12.8f} | {intercept_match}")
    print(f"{ 'Predictions':<15} | {'[Combined]':<12} | {'[Combined]':<12} | {pred_match}")
    print("-" * 60)

    if coef_match and intercept_match and pred_match:
        print("\nRESULT: SUCCESS! Our implementation matches sklearn to 1e-10 precision.")
    else:
        print("\nRESULT: FAILURE! There are significant differences between implementations.")

if __name__ == "__main__":
    compare_implementations()
