import pandas as pd

def calculate_csv_match_percentage(file1_path, file2_path):
    # Load CSV files into DataFrames
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    # Check if dimensions match
    if df1.shape != df2.shape:
        print(f"Warning: Shapes do not match! File 1: {df1.shape}, File 2: {df2.shape}")
        # Align DataFrames to the smallest common shape for direct comparison
        min_rows = min(len(df1), len(df2))
        common_cols = df1.columns.intersection(df2.columns)
        df1 = df1.loc[:min_rows-1, common_cols]
        df2 = df2.loc[:min_rows-1, common_cols]
    
    # Calculate element-by-element equality (handles NaN values properly)
    matches = (df1 == df2) | (df1.isna() & df2.isna())
    
    # Calculate total elements and percentage match
    total_elements = df1.size
    matching_elements = matches.sum().sum()
    match_percentage = (matching_elements / total_elements) * 100
    
    return match_percentage

# Example Usage
file1 = "C:\\Users\\saina\\OneDrive\\Desktop\\QSED\\outputs\\my_experiment\\e3\\entropy_matrix_full.csv"
file2 = "C:\\Users\\saina\\OneDrive\\Desktop\\QSED\\outputs\\my_experiment\\e1\\entropy_matrix_full.csv"
percentage = calculate_csv_match_percentage(file1, file2)
print(f"Match Percentage: {percentage:.2f}%")