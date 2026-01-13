import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.r_squared = None
        self.adj_r_squared = None
        self.standard_errors = None
        self.t_statistics = None
        self.p_values = None
        self._is_fitted = False
        self.n_samples = 0
        self.n_features = 0

    def fit(self, X, y):
        """
        Fit the linear regression model using Ordinary Least Squares.
        
        Parameters:
        X (np.array): Feature matrix of shape (n_samples, n_features)
        y (np.array): Target vector of shape (n_samples,)
        """
        # Ensure inputs are numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Validation
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty dataset provided.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_samples, self.n_features = X.shape
        
        if self.n_samples <= self.n_features:
             # Handle too few samples gracefully but allow fit to attempt
             pass 

        # Add column of ones for intercept
        X_with_intercept = np.column_stack((np.ones(self.n_samples), X))
        
        # Calculate coefficients using OLS closed form: (X^T X)^-1 X^T y
        # We use pinv for stability in case of singular matrix
        xtx = np.dot(X_with_intercept.T, X_with_intercept)
        xtx_inv = np.linalg.pinv(xtx)
        xty = np.dot(X_with_intercept.T, y)
        
        beta = np.dot(xtx_inv, xty)
        
        # Separate intercept and coefficients
        self.intercept = beta[0]
        self.coefficients = beta[1:]
        
        # Calculate predictions for training data
        y_pred = np.dot(X_with_intercept, beta)
        
        # Calculate statistics
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        
        # R-squared
        self.r_squared = 1 - (ss_res / ss_tot)
        
        # Adjusted R-squared
        # 1 - (1-R^2)(n-1)/(n-p-1)
        self.adj_r_squared = 1 - (1 - self.r_squared) * (self.n_samples - 1) / (self.n_samples - self.n_features - 1)
        
        # Variance of the error term (sigma squared)
        # Degrees of freedom = n - p - 1 (where p is n_features)
        degrees_of_freedom = self.n_samples - self.n_features - 1
        
        if degrees_of_freedom > 0:
            sigma_squared = ss_res / degrees_of_freedom
            
            # Variance-covariance matrix of coefficients
            var_cov_matrix = sigma_squared * xtx_inv
            
            # Standard errors are square root of diagonal elements
            self.standard_errors = np.sqrt(np.diagonal(var_cov_matrix))
            
            # Calculate t-statistics
            self.t_statistics = beta / self.standard_errors
            
            # Calculate p-values (two-tailed test)
            # survival function (sf) is 1 - cdf
            self.p_values = stats.t.sf(np.abs(self.t_statistics), degrees_of_freedom) * 2
            
        else:
            self.standard_errors = np.full(self.n_features + 1, np.nan)
            self.t_statistics = np.full(self.n_features + 1, np.nan)
            self.p_values = np.full(self.n_features + 1, np.nan)
            
        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Parameters:
        X (np.array): Feature matrix of shape (n_samples, n_features)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Add column of ones for intercept
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        
        # Reconstruct full beta (intercept + coeffs)
        beta = np.concatenate(([self.intercept], self.coefficients))
        
        return np.dot(X_with_intercept, beta)

    def plot_fit(self, X, y, filename='regression_plot.png'):
        """
        Create a scatter plot with the fitted regression line.
        Only works for univariate regression (1 feature).
        
        Parameters:
        X (np.array): Feature data (n_samples, 1) or (n_samples,)
        y (np.array): Target data (n_samples,)
        filename (str): Path to save the plot
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before plotting")

        X = np.array(X)
        y = np.array(y)
        
        # Check dimensionality
        if X.ndim > 1 and X.shape[1] > 1:
            raise ValueError("plot_fit only supports univariate regression (1 feature)")
            
        # Flatten X for plotting if needed
        X_flat = X.flatten()
        
        # Get predictions
        y_pred = self.predict(X)
        
        plt.figure(figsize=(10, 6))
        
        # Scatter plot of actual data
        plt.scatter(X_flat, y, color='blue', alpha=0.5, label='Data Points')
        
        # Regression line
        # Sort X for clean line plotting
        sort_idx = np.argsort(X_flat)
        plt.plot(X_flat[sort_idx], y_pred[sort_idx], color='red', linewidth=2, label='Regression Line')
        
        # Add equation text
        equation = f'y = {self.coefficients[0]:.2f}x + {self.intercept:.2f}'
        r2_text = f'R² = {self.r_squared:.4f}'
        
        plt.title(f'Linear Regression Fit\n{equation}, {r2_text}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved to {filename}")

    def summary(self):
        """
        Print a formatted table showing regression statistics.
        Similar to statsmodels output.
        """
        if not self._is_fitted:
            print("Model not fitted.")
            return

        print("=" * 78)
        print("                            OLS Regression Results                            ")
        print("=" * 78)
        print(f"Dep. Variable:                      y   R-squared:                       {self.r_squared:.3f}")
        print(f"Model:                            OLS   Adj. R-squared:                  {self.adj_r_squared:.3f}")
        print(f"No. Observations:                {self.n_samples:<4}   ")
        print("=" * 78)
        print(f"{ '':<15} {'coef':<10} {'std err':<10} {'t':<10} {'P>|t|':<10} {'[0.025':<10} {'0.975]':<10}")
        print("-" * 78)

        # Intercept
        ci_lower = self.intercept - 1.96 * self.standard_errors[0]
        ci_upper = self.intercept + 1.96 * self.standard_errors[0]
        print(f"{ 'const':<15} {self.intercept:<10.4f} {self.standard_errors[0]:<10.3f} {self.t_statistics[0]:<10.3f} {self.p_values[0]:<10.3f} {ci_lower:<10.3f} {ci_upper:<10.3f}")

        # Coefficients
        for i, coef in enumerate(self.coefficients):
            se = self.standard_errors[i+1]
            t = self.t_statistics[i+1]
            p = self.p_values[i+1]
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
            print(f"{ f'x{i+1}':<15} {coef:<10.4f} {se:<10.3f} {t:<10.3f} {p:<10.3f} {ci_lower:<10.3f} {ci_upper:<10.3f}")
            
        print("=" * 78)