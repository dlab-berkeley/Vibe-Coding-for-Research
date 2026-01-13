import pytest
import numpy as np
from linear_regression import LinearRegression

def test_perfect_linear_relationship():
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X.flatten() + 5
    model = LinearRegression().fit(X, y)
    assert np.isclose(model.r_squared, 1.0)
    assert np.isclose(model.coefficients[0], 2.0)
    assert np.isclose(model.intercept, 5.0)

def test_random_noise_r2():
    np.random.seed(42)
    X = np.random.normal(0, 1, (1000, 1))
    y = np.random.normal(0, 1, 1000)
    model = LinearRegression().fit(X, y)
    assert model.r_squared < 0.05 # Should be very low

def test_single_data_point():
    X = np.array([[1]])
    y = np.array([2])
    model = LinearRegression()
    # Should either raise or result in nan/inf statistics, but not crash
    with pytest.warns(RuntimeWarning): # Expected for division by zero in stats
         model.fit(X, y)

def test_empty_dataset():
    X = np.array([])
    y = np.array([])
    model = LinearRegression()
    with pytest.raises(ValueError, match="Empty dataset"):
        model.fit(X, y)

def test_constant_y():
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.full(100, 5)
    model = LinearRegression().fit(X, y)
    assert np.isclose(model.coefficients[0], 0.0, atol=1e-10)
    assert np.isclose(model.intercept, 5.0)

def test_constant_x():
    X = np.full((100, 1), 5)
    y = np.linspace(0, 10, 100)
    model = LinearRegression().fit(X, y)
    # With pinv, this shouldn't crash. Slope usually defaults to 0 or nan
    assert model._is_fitted

def test_negative_slope():
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = -3 * X.flatten() + 10
    model = LinearRegression().fit(X, y)
    assert np.isclose(model.coefficients[0], -3.0)

def test_large_dataset():
    n = 100000
    X = np.random.normal(0, 1, (n, 1))
    y = 2 * X.flatten() + 5 + np.random.normal(0, 1, n)
    model = LinearRegression().fit(X, y)
    assert model._is_fitted
    assert np.isclose(model.coefficients[0], 2.0, atol=0.05)

def test_prediction_accuracy():
    X_train = np.array([[1], [2], [3]])
    y_train = np.array([2, 4, 6])
    model = LinearRegression().fit(X_train, y_train)
    
    X_new = np.array([[4], [5]])
    y_pred = model.predict(X_new)
    assert np.allclose(y_pred, [8, 10])
