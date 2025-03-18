import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from spellchecker import SpellChecker
import warnings
import math
import os
import missingno as msno


import difflib



# Suppress all warnings
warnings.filterwarnings('ignore')


def print_section_header(title):
    """Prints a formatted section header with dashes."""
    print("\n" + "-" * 50)
    print(f"{title.center(50)}")
    print("-" * 50 + "\n")

def load_data(file_path):
    """Load data from different formats using polars."""
    print_section_header("Loading Data")
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.tsv'):
        return pd.read_csv(file_path, separator='\t')
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        raise ValueError("Polars does not support direct Excel file reading. Convert to CSV or Parquet.")
    else:
        raise ValueError("Unsupported file format")

def check_data_types(df):
    """Check the data types of columns."""
    print_section_header("Checking Data Types")
    print(df.dtypes)
    return df.dtypes

def replace_symbols_and_convert_to_float(df):
    """Replace symbols and convert to float."""
    print_section_header("Handling Symbols & Conversion")
    problematic_cols = [col for col in df.columns if df[col].astype(str).str.contains(r'[\$,₹,-]', regex=True).any()]
    
    if problematic_cols:
        print("Columns containing $, ₹, -, or , before processing:", problematic_cols)
    df.replace({'-': np.nan, ',': np.nan, '\$': np.nan, '₹': np.nan}, regex=True, inplace=True)
    for col in problematic_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df



def fix_incorrect_data_types(df):
    for col in df.columns:
        try:
            # Attempt to convert column to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            # Log any errors for debugging purposes
            print(f"Could not convert column {col} due to: {e}")
    return df




import difflib

def fix_spelling_errors_in_columns(df):
    """Fix spelling errors in column names by interacting with the user."""
    
    print_section_header("Checking for spelling error in column names")
    for idx, col in enumerate(df.columns, start=1):
        print(f"{idx}. {col}")

    print("\n" + "-" * 40)
    
    incorrect_columns = input("Enter the index of the columns with incorrect spelling (comma-separated), or press Enter to skip: ").strip()

    if not incorrect_columns:
        print("No changes made.")
        return df

    incorrect_columns = incorrect_columns.split(',')
    incorrect_columns = [df.columns[int(i.strip()) - 1] for i in incorrect_columns if i.strip().isdigit()]

    corrected_columns = {}
    
    for col in incorrect_columns:
        suggestion = difflib.get_close_matches(col, df.columns, n=1, cutoff=0.8)
        if suggestion and suggestion[0] != col:
            print(f"Suggested correction for '{col}': {suggestion[0]}")

        correct_spelling = input(f"Enter the correct spelling for '{col}' (or press Enter to keep it unchanged): ").strip()
        
        if correct_spelling:
            corrected_columns[col] = correct_spelling

    # Rename columns
    df.rename(columns=corrected_columns, inplace=True)
    
    print("Updated Column Names:")
    print("-" * 40)
    print(df.columns.tolist())
    
    return df


import difflib

def fix_spelling_errors_in_categorical(df):
    """Fix spelling errors in categorical columns by asking the user to review unique values."""
    
    
    print_section_header("Checking for Spelling Errors in Categorical Columns")

    
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        unique_values = df[col].dropna().unique()
        
        if not unique_values.any():
            continue  # Skip if no values exist
        
        print(f"\nColumn: {col}")
        for idx, val in enumerate(unique_values, start=1):
            print(f"{idx}. {val}")
        
        incorrect_values = input(
            f"\nEnter the numbers of the incorrect values in '{col}' (comma-separated), or press Enter to skip: "
        ).strip()

        if not incorrect_values:
            continue
        
        incorrect_values = incorrect_values.split(',')
        incorrect_values = [unique_values[int(i.strip()) - 1] for i in incorrect_values if i.strip().isdigit()]

        corrected_values = {}
        
        for value in incorrect_values:
            suggestion = difflib.get_close_matches(value, unique_values, n=1, cutoff=0.8)
            if suggestion:
                print(f"Suggested correction for '{value}': {suggestion[0]}")

            correct_spelling = input(f"Enter the correct spelling for '{value}' (or press Enter to keep it unchanged): ").strip()
            
            if correct_spelling:
                corrected_values[value] = correct_spelling
        
        # Apply corrections to the column
        df[col] = df[col].replace(corrected_values)

    print("\nSpelling correction process completed.")
    return df



import numpy as np

def handle_negative_values(df):
    """Handle negative values by printing column names with negatives and replacing them with absolute values."""
    print_section_header("Checking for negative values")

    # Iterate over each numerical column
    
    for col in df.select_dtypes(include=[np.number]).columns:
        # Check the minimum value of the column
        min_val = df[col].min()
        
        # If the minimum value is negative, print the column name
        if min_val < 0:
            print(f"Column '{col}' contains negative values.")
            
            # Apply absolute function to the column
            df[col] = df[col].abs()
    
    return df


def handle_missing_values(df):
    """Handle missing values with visualization."""
    print_section_header("Handling Missing Values")
    missing_columns = df.columns[df.isnull().any()].tolist()
    
    if missing_columns:
        print("Columns containing missing values:", missing_columns)
    else:
        print("No missing values found.")
    
    plt.figure(figsize=(10, 6))
    msno.bar(df)
    
    
    num_df = df.select_dtypes(include=[np.number])
    imputer = KNNImputer(n_neighbors=5)
    df[num_df.columns] = imputer.fit_transform(num_df)
    print("Applied KNN imputation for numerical columns.")
    
    categorical_cols = df.select_dtypes(include=[object, 'category']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
            print(f"Filled missing values in '{col}' with mode: {mode_value}")
    return df

def handle_duplicates(df):
    """Handle duplicate records."""
    print_section_header("Handling Duplicates")
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"Duplicate rows found: {duplicate_count}. Dropping duplicates...")
        df = df.drop_duplicates()
    else:
        print("No duplicate rows found.")
    return df


def check_outliers(df):
    """Check for outliers using IQR method."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outliers

def remove_outliers(df):
    """Remove outliers using IQR method for numerical columns."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Remove outliers for each column separately
        df = df[~((df[col] < lower_bound) | (df[col] > upper_bound))]

    return df.reset_index(drop=True)

    
def check_and_handle_imbalance(df, target_col, output_folder="output"):
    """Check for class imbalance and allow user to handle it using oversampling or undersampling.
       Saves class distribution images in the specified output folder."""
    
    print_section_header("Checking for Data Imbalance")
    
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame.")
    
    class_counts = df[target_col].value_counts()
    min_count, max_count = class_counts.min(), class_counts.max()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    print(f"\nClass Distribution:\n{class_counts}\n")
    
    if imbalance_ratio > 1.5:
        print(f"\U0001F534 The target column '{target_col}' is **imbalanced**.")
    else:
        print(f"\U0001F7E2 The target column '{target_col}' is **balanced**.")
    
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Plot class distribution
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.values, y=class_counts.index, palette="viridis")
    plt.xlabel("Count")
    plt.ylabel("Class")
    plt.title(f"Class Distribution of '{target_col}'")
    
    # Save plot
    plot_path = os.path.join(output_folder, f"class_distribution_{target_col}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\U0001F4C2 Class distribution plot saved at: {plot_path}")
    
    # Ask user if they want to fix imbalance
    fix_imbalance = input("\nDo you want to handle the imbalance? (yes/no): ").strip().lower()
    if fix_imbalance != "yes":
        print("No changes applied.")
        return df

    # Ask for oversampling or undersampling
    choice = input("\nChoose a method:\n1. Oversampling (Random Duplication)\n2. Undersampling (Random Removal)\nEnter your choice (1/2): ").strip()
    
    if choice not in ["1", "2"]:
        print("Invalid choice. No changes applied.")
        return df
    
    print_section_header("Applying Class Balancing...")
    balanced_df = pd.DataFrame(columns=df.columns)  # Create empty DataFrame for resampling
    
    if choice == "1":  # **Oversampling**
        print("✅ Applying **Oversampling**: Duplicating minority class samples.")
        max_samples = max_count  # Target number of samples for each class
        
        for label, count in class_counts.items():
            class_subset = df[df[target_col] == label]
            additional_samples = class_subset.sample(n=max_samples - count, replace=True, random_state=42)
            balanced_df = pd.concat([balanced_df, class_subset, additional_samples], axis=0)
    
    else:  # **Undersampling**
        print("✅ Applying **Undersampling**: Removing excess majority class samples.")
        min_samples = min_count  # Target number of samples for each class
        
        for label, count in class_counts.items():
            class_subset = df[df[target_col] == label]
            class_subset_sampled = class_subset.sample(n=min_samples, random_state=42)
            balanced_df = pd.concat([balanced_df, class_subset_sampled], axis=0)
    
    # Shuffle the dataset after resampling
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save new class distribution plot
    new_class_counts = balanced_df[target_col].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=new_class_counts.values, y=new_class_counts.index, palette="viridis")
    plt.xlabel("Count")
    plt.ylabel("Class")
    plt.title(f"Balanced Class Distribution of '{target_col}'")
    
    balanced_plot_path = os.path.join(output_folder, f"balanced_class_distribution_{target_col}.png")
    plt.savefig(balanced_plot_path)
    plt.close()
    print(f"\U0001F4C2 Balanced class distribution plot saved at: {balanced_plot_path}")
    
    print(f"\n✅ Class distribution after resampling:\n{new_class_counts}")
    return balanced_df


def check_skewness(df):
    """Check skewness in numerical columns."""
    return df.skew()

def fix_skewness(df):
    """Fix skewness in numerical columns using log transformation."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].skew() > 1:
            df[col] = np.log1p(df[col])
    return df

def check_multicollinearity(df, threshold=0.7):
    """Check for multicollinearity using correlation matrix and remove highly correlated features."""
    print_section_header("Checking for Multicollinearity problem")

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    num_df = df.select_dtypes(include=['int', 'float'])
    correlation_matrix = num_df.corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Matrix")
    
    plot_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Correlation matrix plot saved as '{plot_path}'")
    
    # Identify highly correlated features and remove them
    to_drop = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):  # Avoid duplicate pairs
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                col_to_drop = correlation_matrix.columns[i]
                to_drop.add(col_to_drop)
    
    if to_drop:
        print(f"Dropping highly correlated columns: {', '.join(to_drop)}")
        df.drop(columns=to_drop, inplace=True)
    else:
        print("No highly correlated feature pairs found above the threshold.")
    
    return df



def check_cardinality(df):
    """
    Check the cardinality (number of unique values) of categorical columns and remove columns with only one unique value.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame after removing low-cardinality columns.
        pd.Series: A series with column names as index and their cardinality as values.
    """
    print_section_header("Checking for Cardinality")

    # Select only categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    print(f"Categorical columns found: {list(categorical_cols)}")

    # Calculate cardinality
    cardinality = df[categorical_cols].nunique()
    print(f"Cardinality of categorical columns:\n{cardinality}")

    # Remove columns with only one unique value
    low_cardinality_cols = cardinality[cardinality == 1].index
    if len(low_cardinality_cols) > 0:
        print(f"\nRemoving columns with only one unique value: {list(low_cardinality_cols)}")
        df = df.drop(columns=low_cardinality_cols)
    else:
        print("\nNo columns removed beacyuse there are no low cardinality column")

    return df, cardinality


def save_cleaned_data(df, file_name="cleaned_data.csv"):
    """Save cleaned DataFrame to a CSV file."""
    print_section_header("Saving cleaned data")

    df.to_csv(file_name, index=False)
    print(f"Cleaned data saved to {file_name}")


def save_boxplots(df, output_filename="output/boxplots.png"):
    """
    Create boxplots for numerical columns in a DataFrame and save the plot as a PNG file.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        output_filename (str): The filename for saving the boxplots image.
    """
    print_section_header("Checking for outliers")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    print(f"Numerical columns found: {list(numerical_cols)}")
    
    if numerical_cols.empty:
        print("No numerical columns found in the DataFrame.")
        return
    
    # Determine the number of rows and columns for subplots
    num_cols = len(numerical_cols)
    cols_per_row = 3
    num_rows = math.ceil(num_cols / cols_per_row)
    
    # Create subplots
    fig, axes = plt.subplots(num_rows, min(num_cols, cols_per_row), figsize=(15, 5 * num_rows))
    axes = axes.flatten() if num_cols > 1 else [axes]
    
    # Plot boxplots
    for i, col in enumerate(numerical_cols):
        df.boxplot(column=col, ax=axes[i])
        axes[i].set_title(f"Boxplot of {col}")
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Boxplots saved as '{output_filename}'")
    plt.close()



def clean_data(df):
    """Main function to clean the data."""
    df = handle_negative_values(df)
    df = replace_symbols_and_convert_to_float(df)
    df = fix_spelling_errors_in_columns(df)
    df = fix_spelling_errors_in_categorical(df)
    
    df = handle_missing_values(df)
    df = handle_duplicates(df)
    check_cardinality(df)
    save_boxplots(df)
    
    # df = remove_outliers(df)
    df = fix_skewness(df)
    df=check_multicollinearity(df)
    print_section_header("Enter target column")
    target_col = input("Enter the target column: ")
    df=check_and_handle_imbalance(df,target_col)
    df=save_cleaned_data(df)
    return df

