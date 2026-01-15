import pandas as pd
import sys
import os
import shutil

# Ensure we can import the class from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from linear_regression import LinearRegression

def main():
    # Paths
    data_path = os.path.join('Data', 'productivity_study.csv')
    results_dir = os.path.join('livedemo', 'results')
    summary_path = os.path.join(results_dir, 'summary.txt')
    plot_target_path = os.path.join(results_dir, 'regression_plot.png')

    # 1. Load the data
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return

    X = df[['Coffee_Cups']].values
    y = df['Productivity_Score'].values

    # 2. Import and use LinearRegression class
    print("Fitting model...")
    model = LinearRegression()
    model.fit(X, y)

    # 3. Save the model summary
    print(f"Saving summary to {summary_path}...")
    with open(summary_path, 'w') as f:
        # Redirect stdout to the file to capture print statements from summary()
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            model.summary()
        finally:
            sys.stdout = original_stdout

    # 4. Save the plot
    # plot_fit saves 'regression_plot.png' in the current working directory
    print("Generating plot...")
    model.plot_fit(X, y)
    
    # Move the file
    if os.path.exists('regression_plot.png'):
        print(f"Moving plot to {plot_target_path}...")
        if os.path.exists(plot_target_path):
            os.remove(plot_target_path)
        shutil.move('regression_plot.png', plot_target_path)
    else:
        print("Error: plot_fit() did not create 'regression_plot.png' in the current directory.")

    print("Analysis complete.")

if __name__ == "__main__":
    main()
