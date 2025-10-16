# CE 49X - Lab 2: Soil Test Data Analysis

# Student Name: _Yusuf________
# Student ID: ___Yalçınkaya____
# Date: ___14/10/2025________

import pandas as pd
import numpy as np

file_path = 'soil_test.csv'
def load_data(file_path):
    """
    Load the soil test dataset from a CSV file.
    
    Parameters:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame, or None if the file is not found.
    """
    # TODO: Implement data loading with error handling

    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.\n")
        return df
    except FileNotFoundError:
        print("Error: File not found.")
        return None

    pass

def clean_data(df):
    """
    Clean the dataset by handling missing values and removing outliers from 'soil_ph'.
    
    For each column in ['soil_ph', 'nitrogen', 'phosphorus', 'moisture']:
    - Missing values are filled with the column mean.
    
    Additionally, remove outliers in 'soil_ph' that are more than 3 standard deviations from the mean.
    
    Parameters:
        df (pd.DataFrame): The raw DataFrame.
        
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df_cleaned = df.copy()
    cols = ['soil_ph', 'nitrogen', 'phosphorus', 'moisture']
    
    # TODO: Fill missing values in each specified column with the column mean
    for col in cols:
        mean_val = df_cleaned[col].mean()
        df_cleaned[col] = df_cleaned[col].fillna(mean_val)
    
    # TODO: Remove outliers in 'soil_ph': values more than 3 standard deviations from the mean
    mean_ph = df_cleaned['soil_ph'].mean()
    std_ph = df_cleaned['soil_ph'].std()
    df_cleaned = df_cleaned[
        (df_cleaned['soil_ph'] > mean_ph - 3 * std_ph) &
        (df_cleaned['soil_ph'] < mean_ph + 3 * std_ph)
        ]
    
    print(df_cleaned.head())
    return df_cleaned

def compute_statistics(df, column):
    """
    Compute and print descriptive statistics for the specified column.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column for which to compute statistics.
    """
    # TODO: Calculate minimum value
    min_val = df[column].min()
    
    # TODO: Calculate maximum value
    max_val = df[column].max()
    
    # TODO: Calculate mean value
    mean_val = df[column].mean()
    
    # TODO: Calculate median value
    median_val = df[column].median()
    
    # TODO: Calculate standard deviation
    std_val = df[column].std()
    
    print(f"\nDescriptive statistics for '{column}':")
    print(f"  Minimum: {min_val}")
    print(f"  Maximum: {max_val}")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Median: {median_val:.2f}")
    print(f"  Standard Deviation: {std_val:.2f}")

def main():
    # TODO: Update the file path to point to your soil_test.csv file
    file_path = 'soil_test.csv'  # Update this path as needed

    
    # TODO: Load the dataset using the load_data function
    df = load_data(file_path)
    
    # TODO: Clean the dataset using the clean_data function
    if df is not None:
        df_clean = clean_data(df)
    
    # TODO: Compute and display statistics for the 'soil_ph' column
    compute_statistics(df_clean, 'soil_ph')

    # TODO: (Optional) Compute statistics for other columns
    # compute_statistics(df_clean, 'nitrogen')
    compute_statistics(df_clean, 'nitrogen')
    # compute_statistics(df_clean, 'phosphorus')
    compute_statistics(df_clean, 'phosphorus')
    # compute_statistics(df_clean, 'moisture')
    compute_statistics(df_clean, 'moisture')


if __name__ == '__main__':
    main()

# =============================================================================
# REFLECTION QUESTIONS
# =============================================================================
# Answer these questions in comments below:

# 1. What was the most challenging part of this lab?
# Answer: 
#The most challenging part was implementing proper data cleaning and ensuring 
# that missing values and outliers were handled correctly. Understanding how to 
# detect and remove outliers based on standard deviation thresholds required 
# careful thinking, especially to avoid accidentally removing valid data.

# 2. How could soil data analysis help civil engineers in real projects?
# Answer:
# Soil data analysis is crucial for civil engineers because it helps determine 
# the suitability of a site for construction. Parameters like soil pH, moisture, 
# and nutrient content affect soil stability, foundation design, and drainage 
# characteristics. By analyzing these properties, engineers can predict soil 
# behavior under loads and design safer, more sustainable structures.

# 3. What additional features would make this soil analysis tool more useful?
# Answer: 
#Adding data visualization (such as histograms, box plots, and scatter plots) 
# would make it easier to identify trends and anomalies. Implementing 
# correlation analysis between variables and exporting summarized reports 
# could also improve its usefulness for decision-making in real projects.

# 4. How did error handling improve the robustness of your code?
# Answer: 
# Error handling made the program more robust by preventing it from crashing 
# when the CSV file was missing or incorrectly named. Instead of stopping 
# execution with an error, the code displays a clear message and safely exits, 
# making it more user-friendly and reliable in real-world scenarios