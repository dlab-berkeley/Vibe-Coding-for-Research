import numpy as np
import pandas as pd
from livedemo.linear_regression import LinearRegression

def generate_data(slope, intercept, noise_sd, n_samples):
    X = np.random.randn(n_samples, 1)
    noise = np.random.normal(0, noise_sd, n_samples)
    y = slope * X.flatten() + intercept + noise
    return X, y

def test_sample_size_effect():
    print("\n--- Test 1: Sample Size Effect (SE should halve when n quadruples) ---")
    slope, intercept, noise_sd = 2, 5, 1
    sample_sizes = [100, 400, 1600]
    std_errors = []

    print(f"{ 'n':<8} {'Slope SE':<12} {'Ratio (prev/curr)':<18} {'Expected Ratio':<15}")
    print("-" * 60)

    for i, n in enumerate(sample_sizes):
        X, y = generate_data(slope, intercept, noise_sd, n)
        model = LinearRegression()
        model.fit(X, y)
        se = model.standard_errors[1] # Slope SE
        std_errors.append(se)
        
        ratio_str = "-"
        expected_str = "-"
        if i > 0:
            ratio = std_errors[i-1] / se
            ratio_str = f"{ratio:.4f}"
            expected_str = f"{np.sqrt(sample_sizes[i]/sample_sizes[i-1]):.4f}" # sqrt(4) = 2
            
        print(f"{n:<8} {se:<12.4f} {ratio_str:<18} {expected_str:<15}")

def test_noise_variance_effect():
    print("\n--- Test 2: Noise Variance Effect (SE should scale with noise_sd) ---")
    slope, intercept, n = 2, 5, 1000
    noise_levels = [0.5, 1.0, 2.0]
    std_errors = []

    print(f"{ 'Noise SD':<10} {'Slope SE':<12} {'Ratio (curr/prev)':<18} {'Expected Ratio':<15}")
    print("-" * 60)

    for i, noise_sd in enumerate(noise_levels):
        X, y = generate_data(slope, intercept, noise_sd, n)
        model = LinearRegression()
        model.fit(X, y)
        se = model.standard_errors[1]
        std_errors.append(se)

        ratio_str = "-"
        expected_str = "-"
        if i > 0:
            ratio = se / std_errors[i-1]
            ratio_str = f"{ratio:.4f}"
            expected_str = f"{noise_levels[i]/noise_levels[i-1]:.4f}" # 2.0
            
        print(f"{noise_sd:<10} {se:<12.4f} {ratio_str:<18} {expected_str:<15}")

def test_duplicated_sample():
    print("\n--- Test 3: Duplicated Sample (SE should decrease by 1/sqrt(2)) ---")
    slope, intercept, noise_sd, n = 2, 5, 1.0, 500
    
    # Original Data
    X, y = generate_data(slope, intercept, noise_sd, n)
    
    # Fit Original
    model_orig = LinearRegression()
    model_orig.fit(X, y)
    se_orig = model_orig.standard_errors[1]
    
    # Duplicate Data
    X_dup = np.vstack([X, X])
    y_dup = np.hstack([y, y])
    
    # Fit Duplicated
    model_dup = LinearRegression()
    model_dup.fit(X_dup, y_dup)
    se_dup = model_dup.standard_errors[1]
    
    ratio = se_dup / se_orig
    expected_ratio = 1 / np.sqrt(2)
    
    print(f"Original n={n}, Slope SE: {se_orig:.6f}")
    print(f"Duplicated n={2*n}, Slope SE: {se_dup:.6f}")
    print(f"Ratio (Dup/Orig): {ratio:.6f}")
    print(f"Expected Ratio:   {expected_ratio:.6f}")
    
    # Check if close
    if np.isclose(ratio, expected_ratio, atol=0.01):
         print("SUCCESS: SE decreased by approx 1/sqrt(2)")
    else:
         print("FAILURE: SE scaling mismatch")


if __name__ == "__main__":
    test_sample_size_effect()
    test_noise_variance_effect()
    test_duplicated_sample()
