import numpy as np
from linear_regression import LinearRegression

def generate_data(n, slope=2.0, intercept=5.0, noise_sd=1.0, seed=42):
    np.random.seed(seed)
    X = np.random.normal(0, 1, n)
    y = slope * X + intercept + np.random.normal(0, noise_sd, n)
    return X, y

def run_theory_tests():
    print("=" * 60)
    print("THEORETICAL VALIDATION TESTS")
    print("=" * 60)

    # --- Test 1: Sample Size Effect (1/sqrt(n)) ---
    print("\nTest 1: Sample Size Effect (Expected: SE halves when N quadruples)")
    n_values = [100, 400, 1600]
    prev_se = None
    
    for n in n_values:
        X, y = generate_data(n=n)
        model = LinearRegression().fit(X, y)
        current_se = model.standard_errors[1] # Slope SE
        
        if prev_se is not None:
            ratio = current_se / prev_se
            print(f"N: {n:<5} | SE: {current_se:.6f} | Ratio: {ratio:.3f} (Target: 0.500)")
        else:
            print(f"N: {n:<5} | SE: {current_se:.6f} | Base Case")
        prev_se = current_se

    # --- Test 2: Noise Variance Effect (Linear scaling) ---
    print("\nTest 2: Noise Variance Effect (Expected: SE scales with Noise SD)")
    noise_values = [0.5, 1.0, 2.0]
    prev_se = None
    
    for sd in noise_values:
        X, y = generate_data(n=1000, noise_sd=sd)
        model = LinearRegression().fit(X, y)
        current_se = model.standard_errors[1]
        
        if prev_se is not None:
            ratio = current_se / prev_se
            print(f"SD: {sd:<5} | SE: {current_se:.6f} | Ratio: {ratio:.3f} (Target: 2.000)")
        else:
            print(f"SD: {sd:<5} | SE: {current_se:.6f} | Base Case")
        prev_se = current_se

    # --- Test 3: Duplicated Sample ---
    print("\nTest 3: Duplicated Sample (Expected: SE should decrease by exactly 1/sqrt(2) ≈ 0.707)")
    X_orig, y_orig = generate_data(n=500)
    model_orig = LinearRegression().fit(X_orig, y_orig)
    se_orig = model_orig.standard_errors[1]
    
    X_dup = np.tile(X_orig, 2)
    y_dup = np.tile(y_orig, 2)
    model_dup = LinearRegression().fit(X_dup, y_dup)
    se_dup = model_dup.standard_errors[1]
    
    ratio = se_dup / se_orig
    target = 1 / np.sqrt(2)
    
    print(f"Original (N=500)   SE: {se_orig:.6f}")
    print(f"Duplicated (N=1000) SE: {se_dup:.6f}")
    print(f"Ratio: {ratio:.4f} (Target: {target:.4f})")
    
    if np.isclose(ratio, target, atol=1e-3):
        print("RESULT: PASS - SE follows the OLS matrix identity for duplicated data.")
    else:
        print("RESULT: FAIL - SE does not follow the expected 1/sqrt(2) scaling.")

if __name__ == "__main__":
    run_theory_tests()
