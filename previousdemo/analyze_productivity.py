import pandas as pd
import sys
import os
from linear_regression import LinearRegression

# 1. Load the data
# Adjust path to be relative to the script location or current working directory
data_path = os.path.join('..', 'Data', 'productivity_study.csv')
df = pd.read_csv(data_path)

# 2. Model 'Productivity_Score' based on 'Coffee_Cups'
X = df[['Coffee_Cups']].values
y = df['Productivity_Score'].values

model = LinearRegression()
model.fit(X, y)

# 3. Save the model summary to 'livedemo/results/summary.txt'
# Ensure results directory exists
os.makedirs('results', exist_ok=True)
summary_path = os.path.join('results', 'summary.txt')

with open(summary_path, 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f
    try:
        model.summary()
    finally:
        sys.stdout = original_stdout

# 4. Save the plot to 'livedemo/results/regression_plot.png'
plot_path = os.path.join('results', 'regression_plot.png')
model.plot_fit(X, y, filename=plot_path)

print(f"Analysis complete. Results saved to {summary_path} and {plot_path}")
