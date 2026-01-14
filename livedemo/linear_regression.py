import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.r_squared = None
        self.standard_errors = None
        self._beta = None  # Store full beta vector (intercept + coeffs)

    def fit(self, X, y):
        """
        Fits the linear regression model using Ordinary Least Squares.
        
        Parameters:
        X (np.array): Feature matrix of shape (n_samples, n_features)
        y (np.array): Target values of shape (n_samples,)
        """
        # Ensure inputs are numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        self.n_samples = n_samples
        self.n_features = n_features
        
        # Add a column of ones for the intercept
        X_with_intercept = np.c_[np.ones(n_samples), X]
        
        # OLS Closed Form Solution: beta = (X.T * X)^-1 * X.T * y
        try:
            xtx_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is singular and cannot be inverted. Features might be collinear.")
            
        self._beta = xtx_inv @ X_with_intercept.T @ y
        
        # Separate intercept and coefficients
        self.intercept = self._beta[0]
        self.coefficients = self._beta[1:]
        
        # --- Statistics Calculation ---
        
        # Predictions on training data
        y_pred = X_with_intercept @ self._beta
        
        # Residuals
        residuals = y - y_pred
        
        # Sum of Squared Errors (SSE)
        sse = np.sum(residuals**2)
        
        # Total Sum of Squares (SST)
        sst = np.sum((y - np.mean(y))**2)
        
        # R-squared
        self.r_squared = 1 - (sse / sst) if sst != 0 else 0.0
        
        # Variance of residuals (MSE)
        # Degrees of freedom = n_samples - number of parameters (features + 1 for intercept)
        df_resid = n_samples - (n_features + 1)
        
        if df_resid > 0:
            residual_variance = sse / df_resid
            
            # Covariance matrix of beta
            beta_cov_matrix = residual_variance * xtx_inv
            
            # Standard errors are the square root of the diagonal elements
            self.standard_errors = np.sqrt(np.diag(beta_cov_matrix))
            
            # Separate intercept SE from coefficient SEs if needed, 
            # but usually standard_errors refers to all params.
            # Here we follow the structure of coefficients vs intercept, 
            # so we might want to split them or keep them as a vector.
            # The prompt asks for "standard_errors" property. 
            # Often this implies a vector matching coefficients, but let's store the full vector
            # to be safe, or clarify. I'll store the full vector corresponding to [intercept, coeffs...]
            # effectively.
        else:
            self.standard_errors = np.full(n_features + 1, np.nan)

    def predict(self, X):
        """
        Predicts target values for given input features.
        
        Parameters:
        X (np.array): Feature matrix of shape (n_samples, n_features)
        
        Returns:
        np.array: Predicted values
        """
        if self._beta is None:
            raise RuntimeError("Model has not been fitted yet.")
            
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Add a column of ones for the intercept
        X_with_intercept = np.c_[np.ones(n_samples), X]
        
        return X_with_intercept @ self._beta

    def plot_fit(self, X, y):
        """
        Creates a scatter plot of the data points and overlays the fitted regression line.
        Saves the plot as regression_plot.png.
        
        Parameters:
        X (np.array): Feature matrix.
        y (np.array): Target values.
        """
        if self._beta is None:
            raise RuntimeError("Model has not been fitted yet.")
            
        X = np.array(X)
        y = np.array(y)
        
        # Check dimensions for plotting
        if X.ndim == 1:
            X_plot = X
        elif X.shape[1] == 1:
            X_plot = X.flatten()
        else:
            print("Warning: Multiple features detected. Plotting against the first feature.")
            X_plot = X[:, 0]
            
        y_pred = self.predict(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X_plot, y, color='blue', label='Data Points', alpha=0.7)
        plt.plot(X_plot, y_pred, color='red', linewidth=2, label='Regression Line')
        
        # Equation text (showing first coefficient for simplicity)
        m = self.coefficients[0] if len(self.coefficients) > 0 else 0
        b = self.intercept
        equation = f"y = {m:.2f}x + {b:.2f}"
        
        plt.title(f"Linear Regression Fit\n{equation}, $R^2$ = {self.r_squared:.4f}")
        plt.xlabel("Feature")
        plt.ylabel("Target")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig("regression_plot.png")
        plt.close()
        print("Plot saved as regression_plot.png")

    def summary(self):
        """
        Prints a formatted table showing regression statistics.
        """
        if self._beta is None:
            raise RuntimeError("Model has not been fitted yet.")
            
        # Calculate t-statistics and p-values
        t_stats = self._beta / self.standard_errors
        
        # Degrees of freedom for residuals
        df_resid = self.n_samples - (self.n_features + 1)
        
        # p-values (two-tailed)
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))
        
        # Adjusted R-squared
        adj_r_squared = 1 - (1 - self.r_squared) * (self.n_samples - 1) / df_resid
        
        print("==============================================================================")
        print("                            OLS Regression Results                            ")
        print("==============================================================================")
        print(f"Dep. Variable:                      y   R-squared:                       {self.r_squared:.3f}")
        print(f"Model:                            OLS   Adj. R-squared:                  {adj_r_squared:.3f}")
        print(f"No. Observations:                {self.n_samples:4d}   MSE:                          {np.mean(self.standard_errors**2):.3e}")
        print("==============================================================================")
        print(f"{'':<15} {'coef':>10} {'std err':>10} {'t':>10} {'P>|t|':>10}")
        print("------------------------------------------------------------------------------")
        
        # Intercept
        print(f"{'const':<15} {self.intercept:10.4f} {self.standard_errors[0]:10.4f} {t_stats[0]:10.3f} {p_values[0]:10.3f}")
        
        # Coefficients
        for i, coef in enumerate(self.coefficients):
            print(f"{f'x{i+1}':<15} {coef:10.4f} {self.standard_errors[i+1]:10.4f} {t_stats[i+1]:10.3f} {p_values[i+1]:10.3f}")
            
        print("==============================================================================")
