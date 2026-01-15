import numpy as np
import pandas as pd
from linear_regression import LinearRegression

def generate_data(slope, intercept, noise_sd, n=1000, seed=42):
    np.random.seed(seed)
    X = np.random.normal(0, 1, n)
    noise = np.random.normal(0, noise_sd, n)
    y = slope * X + intercept + noise
    return X, y

def validate_datasets():
    datasets = [
        {"id": 1, "slope": 2.0, "intercept": 5.0, "noise": 1.0, "n": 1000},
        {"id": 2, "slope": -1.5, "intercept": 10.0, "noise": 0.5, "n": 1000},
        {"id": 3, "slope": 0.0, "intercept": 3.0, "noise": 2.0, "n": 1000},
        {"id": 4, "slope": 5.0, "intercept": 0.0, "noise": 1.0, "n": 1000},
        {"id": 5, "slope": 1.0, "intercept": 1.0, "noise": 0.1, "n": 1000},
        {"id": 6, "slope": 1.0, "intercept": 1.0, "noise": 0.0, "n": 1000},
    ]

    results = []

    print(f"{'Dataset':<8} {'True Slope':<12} {'Est Slope':<12} {'Slope SE':<12} {'True Int':<12} {'Est Int':<12} {'Int SE':<12} {'Within 95% CI?'}")
    print("-" * 110)

    for ds in datasets:
        X, y = generate_data(ds["slope"], ds["intercept"], ds["noise"], ds["n"])
        
        model = LinearRegression()
        model.fit(X, y)
        
        est_slope = model.coefficients[0]
        est_intercept = model.intercept
        slope_se = model.standard_errors[1] if model.standard_errors is not None and not np.isnan(model.standard_errors).all() else 0.0
        int_se = model.standard_errors[0] if model.standard_errors is not None and not np.isnan(model.standard_errors).all() else 0.0

        # Validation Logic
        # If noise is 0, SE is 0, so we check for exact match (with float tolerance)
        if ds["noise"] == 0:
            slope_valid = np.isclose(est_slope, ds["slope"])
            int_valid = np.isclose(est_intercept, ds["intercept"])
        else:
            # 95% CI roughly +/- 1.96 SE (using 2 for simplicity as per prompt)
            slope_valid = abs(est_slope - ds["slope"]) <= 2 * slope_se
            int_valid = abs(est_intercept - ds["intercept"]) <= 2 * int_se
            
        is_valid = slope_valid and int_valid
        
        print(f"{ds['id']:<8} {ds['slope']:<12.4f} {est_slope:<12.4f} {slope_se:<12.4f} {ds['intercept']:<12.4f} {est_intercept:<12.4f} {int_se:<12.4f} {'YES' if is_valid else 'NO'}")

if __name__ == "__main__":
    validate_datasets()
