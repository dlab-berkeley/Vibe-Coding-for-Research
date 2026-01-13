import numpy as np
import pandas as pd

def generate_research_data():
    np.random.seed(42)
    n = 100
    
    # Independent Variable: Coffee (0 to 8 cups per day)
    coffee = np.random.normal(3.5, 1.5, n)
    coffee = np.clip(coffee, 0, 8)
    
    # Dependent Variable: Productivity Score (0 to 100)
    # Base productivity of 40 + 6 points per cup + random variation
    base_score = 40
    effect_per_cup = 6.5
    noise = np.random.normal(0, 8, n)
    
    productivity = base_score + (effect_per_cup * coffee) + noise
    productivity = np.clip(productivity, 0, 100) # Keep within realistic bounds
    
    # Create DataFrame
    df = pd.DataFrame({
        'Coffee_Cups': np.round(coffee, 1),
        'Productivity_Score': np.round(productivity, 1)
    })
    
    # Save to CSV
    filename = 'livedemo/productivity_study.csv'
    df.to_csv(filename, index=False)
    print(f"Dataset generated and saved to {filename}")
    print("\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    generate_research_data()
