import numpy as np
import pandas as pd
from livedemo.linear_regression import LinearRegression

def generate_data(slope, intercept, noise_sd, n_samples):
    """
    Generates synthetic linear data: y = slope * x + intercept + noise
    """
    X = np.random.randn(n_samples, 1)  # Random feature
    noise = np.random.normal(0, noise_sd, n_samples)
    y = slope * X.flatten() + intercept + noise
    return X, y

def validate_datasets():
    # Define datasets parameters
    datasets_params = [
        {"id": 1, "slope": 2.0, "intercept": 5.0, "noise_sd": 1.0, "n": 1000},
        {"id": 2, "slope": -1.5, "intercept": 10.0, "noise_sd": 0.5, "n": 1000},
        {"id": 3, "slope": 0.0, "intercept": 3.0, "noise_sd": 2.0, "n": 1000},
        {"id": 4, "slope": 5.0, "intercept": 0.0, "noise_sd": 1.0, "n": 1000},
        {"id": 5, "slope": 1.0, "intercept": 1.0, "noise_sd": 0.1, "n": 1000},
        {"id": 6, "slope": 1.0, "intercept": 1.0, "noise_sd": 0.0, "n": 1000},
    ]

    results = []

    print(f"{'Dataset':<8} {'True Slope':<12} {'Est. Slope':<12} {'Slope CI OK?':<14} {'True Int.':<12} {'Est. Int.':<12} {'Int. CI OK?':<14}")
    print("-" * 90)

    for params in datasets_params:
        # 1. Generate Data
        X, y = generate_data(params["slope"], params["intercept"], params["noise_sd"], params["n"])

        # 2. Fit Model
        model = LinearRegression()
        model.fit(X, y)

        # Get estimates and standard errors
        # Note: model.coefficients is a list/array, taking the first one for single feature
        est_slope = model.coefficients[0]
        est_intercept = model.intercept
        
        # Standard errors: [intercept_se, slope_se]
        se_intercept = model.standard_errors[0]
        se_slope = model.standard_errors[1]

        # 3. Check Confidence Intervals (approx 2 SEs)
        # Check Slope
        slope_lower = est_slope - 2 * se_slope
        slope_upper = est_slope + 2 * se_slope
        slope_ok = slope_lower <= params["slope"] <= slope_upper

        # Check Intercept
        int_lower = est_intercept - 2 * se_intercept
        int_upper = est_intercept + 2 * se_intercept
        int_ok = int_lower <= params["intercept"] <= int_upper
        
        # Handle zero noise case (SE might be nan or 0)
        if params["noise_sd"] == 0:
            slope_ok = np.isclose(est_slope, params["slope"])
            int_ok = np.isclose(est_intercept, params["intercept"])

        # Print row
        print(f"{params['id']:<8} {params['slope']:<12.4f} {est_slope:<12.4f} {str(slope_ok):<14} {params['intercept']:<12.4f} {est_intercept:<12.4f} {str(int_ok):<14}")

        results.append({
            "Dataset": params["id"],
            "True Slope": params["slope"],
            "Est. Slope": est_slope,
            "Slope SE": se_slope,
            "Slope OK": slope_ok,
            "True Intercept": params["intercept"],
            "Est. Intercept": est_intercept,
            "Intercept SE": se_intercept,
            "Intercept OK": int_ok
        })

if __name__ == "__main__":
    validate_datasets()
