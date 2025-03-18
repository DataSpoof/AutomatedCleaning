import polars as pl
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
    print("\n" + "-" * 100)
    print(f"{title.center(100)}")
    print("-" * 100 + "\n")



import pyfiglet
import shutil

def print_header(title):
    """Prints a formatted section header with ASCII art font centered in the terminal."""
    ascii_banner = pyfiglet.figlet_format(title, font="slant")  # Choose a large font
    terminal_width = shutil.get_terminal_size().columns  # Get terminal width

    # Split the banner into lines and center each line
    for line in ascii_banner.split("\n"):
        print(line.center(terminal_width))  



def load_data(file_path):
    """Load data from different formats using polars."""

    print_header("Automated Cleaning")
    print_header("by DataSpoof")
    print_section_header("Loading Data")
    if file_path.endswith('.csv'):
        return pl.read_csv(file_path)
    elif file_path.endswith('.tsv'):
        return pl.read_csv(file_path, separator='\t')
    elif file_path.endswith('.json'):
        return pl.read_json(file_path)
    elif file_path.endswith('.parquet'):
        return pl.read_parquet(file_path)
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
    """Replace symbols and convert to float using Polars."""
    print_section_header("Handling Symbols & Conversion")
    
    # Identify columns containing unwanted symbols
    problematic_cols = [
        col for col in df.columns 
        if df[col].cast(pl.Utf8, strict=False).str.contains(r'[\$,₹,-]', literal=False).any()
    ]
    
    if problematic_cols:
        print("Columns containing $, ₹, -, or , before processing:", problematic_cols)

    # Replace symbols and convert to float
    df = df.with_columns([
        pl.col(col)
        .str.replace_all(r'[\$,₹,-]', '')  # Remove $, ₹, - symbols
        .cast(pl.Float64, strict=False)  # Convert to float (coerce errors)
        .alias(col)
        for col in problematic_cols
    ])
    
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


import polars as pl
import difflib

def fix_spelling_errors_in_columns(df):
    """Fix spelling errors in column names by interacting with the user."""
    
    print("Checking for spelling errors in column names:")
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
    df = df.rename(corrected_columns)
    
    print("Updated Column Names:")
    print("-" * 40)
    print(df.columns)
    
    return df

def fix_spelling_errors_in_categorical(df):
    """Fix spelling errors in categorical columns by asking the user to review unique values."""
    
    print("Checking for Spelling Errors in Categorical Columns...")

    # Identify categorical columns (string columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == pl.Utf8]

    for col in categorical_columns:
        unique_values = df[col].drop_nulls().unique().to_list()
        
        if not unique_values:
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
        df = df.with_columns(pl.col(col).replace(corrected_values).alias(col))

    print("\nSpelling correction process completed.")
    return df


import numpy as np


def handle_negative_values(df):
    """Handle negative values by printing column names with negatives and replacing them with absolute values."""
    print_section_header("Checking for Negative Values")

    # Identify numerical columns in Polars
    numeric_cols = [col for col, dtype in df.schema.items() if dtype in [pl.Float64, pl.Int64, pl.Int32, pl.Float32]]

    # Iterate over numerical columns and check for negatives
    for col in numeric_cols:
        min_val = df[col].min()
        
        if min_val < 0:
            print(f"Column '{col}' contains negative values.")
            
            # Replace negative values with their absolute counterparts
            df = df.with_columns(pl.col(col).abs().alias(col))
    
    return df



def handle_missing_values(df):
    """Handle missing values with visualization."""
    missing_columns = [col for col in df.columns if df[col].null_count() > 0]
    
    if missing_columns:
        print("Columns containing missing values:", missing_columns)
    else:
        print("No missing values found.")
    
    plt.figure(figsize=(10, 6))
    msno.bar(df.to_pandas())

    # Select only numeric columns
    num_df = df.select(pl.col(pl.Float64, pl.Int64))
    
    # Convert to NumPy
    num_array = num_df.to_numpy()

    # Check shape before applying imputer
    print(f"Shape of num_df: {num_array.shape}")

    imputer = KNNImputer(n_neighbors=5)
    imputed_values = imputer.fit_transform(num_array)

    # Check shape after imputation
    print(f"Shape after imputation: {imputed_values.shape}")

    # Ensure we don't go out of bounds
    min_columns = min(len(num_df.columns), imputed_values.shape[1])

    df = df.with_columns([pl.Series(num_df.columns[i], imputed_values[:, i]) for i in range(min_columns)])

    categorical_cols = df.select(pl.col(pl.Utf8, pl.Categorical)).columns
    for col in categorical_cols:
        mode_value = df[col].mode().to_list()[0]
        df = df.with_columns(df[col].fill_null(mode_value))
    
    return df


def handle_duplicates(df):
    """Handle duplicate records."""
    duplicate_count = df.is_duplicated().sum()
    if duplicate_count > 0:
        print(f"Duplicate rows found: {duplicate_count}. Dropping duplicates...")
        df = df.unique()
    else:
        print("No duplicate rows found.")
    return df

def check_outliers(df):
    """Check for outliers using IQR method."""
    numerical_cols = df.select(pl.col(pl.Float64, pl.Int64)).columns
    outliers = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df.filter((df[col] < lower_bound) | (df[col] > upper_bound))
    return outliers

def remove_outliers(df):
    """Remove outliers using IQR method for numerical columns."""
    numerical_cols = df.select(pl.col(pl.Float64, pl.Int64)).columns
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df.filter((df[col] >= lower_bound) & (df[col] <= upper_bound))
    return df



import polars as pl
import polars as pl

def check_and_handle_imbalance(df, target_col):
    """
    Check for class imbalance and handle it using user-selected undersampling or oversampling.
    
    Parameters:
    - df (pl.DataFrame): The input Polars dataframe
    - target_col (str): The name of the target column
    
    Returns:
    - pl.DataFrame: A balanced dataframe
    """
    
    # Original Class Distribution
    class_counts = df[target_col].value_counts().sort("count")
    min_count, max_count = class_counts["count"].min(), class_counts["count"].max()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print("Original Class Distribution:")
    print(class_counts)

    if imbalance_ratio > 1.5:
        print(f"\nThe target column '{target_col}' is **imbalanced**.")
        
        # Ask the user for input
        method = input("\nChoose a balancing method - 'oversampling' or 'undersampling': ").strip().lower()
        
        balanced_df = []

        if method == "oversampling":
            max_samples = max_count
            for label in class_counts[target_col].to_list():
                subset = df.filter(df[target_col] == label)
                additional_samples = subset.sample(n=max_samples - len(subset), with_replacement=True)
                balanced_df.append(subset)
                balanced_df.append(additional_samples)

        elif method == "undersampling":
            min_samples = min_count
            for label in class_counts[target_col].to_list():
                subset = df.filter(df[target_col] == label)
                subset = subset.sample(n=min_samples)  # Undersample to match the smallest class
                balanced_df.append(subset)

        else:
            raise ValueError("\nInvalid method. Please choose 'oversampling' or 'undersampling'.")

        # Combine all balanced samples and shuffle
        df = pl.concat(balanced_df).sample(fraction=1.0, shuffle=True)

        # Print New Class Distribution
        print("\nBalanced Class Distribution:")
        print(df[target_col].value_counts().sort("count"))

    return df


def check_skewness(df):
    """Check skewness in numerical columns."""
    return df.select(pl.col(pl.Float64, pl.Int64)).skew()

def fix_skewness(df):
    """Fix skewness using log transformation."""
    numerical_cols = df.select(pl.col(pl.Float64, pl.Int64)).columns
    for col in numerical_cols:
        if df[col].skew() > 1:
            df = df.with_columns(pl.log(df[col] + 1).alias(col))
    return df

def check_multicollinearity(df, threshold=0.7):
    """Check for multicollinearity using correlation matrix and remove highly correlated features."""
    num_df = df.select(pl.col(pl.Float64, pl.Int64))
    correlation_matrix = num_df.corr()
    
    to_drop = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix[i, j]) > threshold:
                to_drop.add(correlation_matrix.columns[i])
    
    if to_drop:
        print(f"Dropping highly correlated columns: {', '.join(to_drop)}")
        df = df.drop(to_drop)
    else:
        print("No highly correlated features found above the threshold.")
    
    return df



def check_cardinality(df: pl.DataFrame):
    """
    Check the cardinality (number of unique values) of categorical columns and remove columns with only one unique value.

    Parameters:
        df (pl.DataFrame): The input DataFrame.

    Returns:
        pl.DataFrame: The DataFrame after removing low-cardinality columns.
        dict: A dictionary with column names as keys and their cardinality as values.
    """
    print_section_header("Checking for Cardinality")
    
    # Select categorical columns
    categorical_cols = [col for col in df.columns if df[col].dtype in [pl.Utf8, pl.Categorical]]
    print(f"Categorical columns found: {categorical_cols}")
    
    # Calculate cardinality
    cardinality = {col: df[col].n_unique() for col in categorical_cols}
    print(f"Cardinality of categorical columns:\n{cardinality}")
    
    # Remove columns with only one unique value
    low_cardinality_cols = [col for col, count in cardinality.items() if count == 1]
    if low_cardinality_cols:
        print(f"\nRemoving columns with only one unique value: {low_cardinality_cols}")
        df = df.drop(low_cardinality_cols)
    else:
        print("\nNo columns removed because there are no low cardinality columns")
    
    return df, cardinality

def save_cleaned_data(df: pl.DataFrame, file_name="cleaned_data.csv", quantize=True):
    """
    Save cleaned DataFrame to a CSV file with optional quantization for float and integer columns.
    
    Parameters:
    - df (pl.DataFrame): The input Polars dataframe.
    - file_name (str): The name of the CSV file.
    - quantize (bool): Whether to quantize numeric columns (default: True).
    
    Returns:
    - None
    """
    
    print("\n🔹 Saving cleaned data...")

    if quantize:
        # Convert float64 -> float32, int64 -> int32 to reduce size
        float_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in [pl.Float64, pl.Float32]]
        int_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in [pl.Int64, pl.Int32]]

        df = df.with_columns([df[col].cast(pl.Float32) for col in float_cols])
        df = df.with_columns([df[col].cast(pl.Int32) for col in int_cols])

    # Save to CSV
    df.write_csv(file_name)
    print(f"✅ Cleaned data saved to {file_name}")


def save_boxplots(df: pl.DataFrame, output_filename="output/boxplots.png"):
    """
    Create boxplots for numerical columns in a DataFrame and save the plot as a PNG file.
    
    Parameters:
        df (pl.DataFrame): The input DataFrame.
        output_filename (str): The filename for saving the boxplots image.
    """
    print_section_header("Checking for outliers")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Select numerical columns
    numerical_cols = [col for col in df.columns if df[col].dtype in [pl.Int64, pl.Float64]]
    print(f"Numerical columns found: {numerical_cols}")
    
    if not numerical_cols:
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
        axes[i].boxplot(df[col].to_list(), vert=True)
        axes[i].set_title(f"Boxplot of {col}")
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Boxplots saved as '{output_filename}'")
    plt.close()



import os
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

# Define output directory
OUTPUT_DIR = "output/eda/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_plot(fig, filename):
    """Save the plotly figure to the output directory."""
    fig.write_image(os.path.join(OUTPUT_DIR, filename))

def univariate_analysis(df):
    """Perform univariate analysis for numerical and categorical columns."""
    
    # Convert Polars DataFrame to Pandas
    df_pandas = df.to_pandas()
    print_section_header("Performing Graphical Data Analysis")

    print("\n=== Univariate Analysis ===")


    # Select numerical and categorical columns
    num_cols = df.select(pl.col(pl.Float64, pl.Int64)).columns
    cat_cols = df.select(pl.col(pl.Utf8)).columns

    print("Plotting all the numerical column using Histogram")

    # Subplots for numerical columns
    fig_num = make_subplots(rows=len(num_cols), cols=1, subplot_titles=[f"Histogram of {col}" for col in num_cols])
    for i, col in enumerate(num_cols):
        fig = px.histogram(df_pandas, x=col, nbins=30)
        for trace in fig.data:
            fig_num.add_trace(trace, row=i+1, col=1)
    fig_num.update_layout(title="Univariate Analysis - Numerical", height=300 * len(num_cols))
    save_plot(fig_num, "univariate_numerical.png")

    # Subplots for categorical columns
    print("Plotting all the categorical column using barplot")

    fig_cat = make_subplots(rows=len(cat_cols), cols=1, subplot_titles=[f"Category Distribution of {col}" for col in cat_cols])
    for i, col in enumerate(cat_cols):
        value_counts = df_pandas[col].value_counts().reset_index()
        value_counts.columns = [col, "count"]  # Rename columns explicitly
        fig = px.bar(value_counts, x=col, y="count")
        for trace in fig.data:
            fig_cat.add_trace(trace, row=i+1, col=1)
    fig_cat.update_layout(title="Univariate Analysis - Categorical", height=300 * len(cat_cols))
    save_plot(fig_cat, "univariate_categorical.png")

import itertools

def bivariate_analysis(df):
    """Perform bivariate analysis for all numerical and categorical column combinations."""
    
    df_pandas = df.to_pandas()
    num_cols = df.select(pl.col(pl.Float64, pl.Int64)).columns
    cat_cols = df.select(pl.col(pl.Utf8)).columns

    print("\n=== Bivariate Analysis ===")

    # Scatter Plots for All Pairs of Numerical Columns
    print("Plotting scatter plot for All Pairs of Numerical Columns")

    num_combinations = list(itertools.combinations(num_cols, 2))
    if num_combinations:
        fig_scatter = make_subplots(rows=len(num_combinations), cols=1, 
                                    subplot_titles=[f"Scatter: {x} vs {y}" for x, y in num_combinations])
        for i, (x, y) in enumerate(num_combinations):
            #print(f"Generating Scatter Plot for: {x} vs {y}")
            fig = px.scatter(df_pandas, x=x, y=y)
            for trace in fig.data:
                fig_scatter.add_trace(trace, row=i+1, col=1)
        fig_scatter.update_layout(title="Bivariate Analysis - Scatter Plots", height=400 * len(num_combinations))
        save_plot(fig_scatter, "bivariate_scatter_all.png")

    # Histograms for All Numerical vs Categorical Combinations
    print("Plotting Histograms for All Numerical vs Categorical Combinations")

    num_cat_combinations = list(itertools.product(num_cols, cat_cols))
    if num_cat_combinations:
        fig_hist = make_subplots(rows=len(num_cat_combinations), cols=1,
                                 subplot_titles=[f"Histogram: {num} by {cat}" for num, cat in num_cat_combinations])
        for i, (num, cat) in enumerate(num_cat_combinations):
            #print(f"Generating Histogram for: {num} grouped by {cat}")
            fig = px.histogram(df_pandas, x=num, color=cat)
            for trace in fig.data:
                fig_hist.add_trace(trace, row=i+1, col=1)
        fig_hist.update_layout(title="Bivariate Analysis - Numerical vs Categorical", height=400 * len(num_cat_combinations))
        save_plot(fig_hist, "bivariate_num_vs_cat_all.png")

    # Stacked Bar Plots for All Categorical Combinations
    print("Plotting Stacked Bar Plots for All Categorical Combinations")

    cat_combinations = list(itertools.combinations(cat_cols, 2))
    if cat_combinations:
        fig_bar = make_subplots(rows=len(cat_combinations), cols=1,
                                subplot_titles=[f"Stacked Bar: {x} vs {y}" for x, y in cat_combinations])
        for i, (x, y) in enumerate(cat_combinations):
            #print(f"Generating Stacked Bar Plot for: {x} vs {y}")
            fig = px.bar(df_pandas, x=x, color=y)
            for trace in fig.data:
                fig_bar.add_trace(trace, row=i+1, col=1)
        fig_bar.update_layout(title="Bivariate Analysis - Categorical", height=400 * len(cat_combinations))
        save_plot(fig_bar, "bivariate_cat_vs_cat_all.png")

def multivariate_analysis(df):
    """Perform multivariate analysis using correlation heatmap."""
    print("\n=== Multivariate Analysis ===")
    
    df_pandas = df.to_pandas()
    num_cols = df.select(pl.col(pl.Float64, pl.Int64)).columns
    corr_matrix = df_pandas[num_cols].corr()

    print("Plotting correlation matrix ")
    fig_corr = make_subplots(rows=1, cols=1, subplot_titles=["Correlation Heatmap"])
    heatmap = go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index, colorscale='Blues', zmin=-1, zmax=1)
    fig_corr.add_trace(heatmap)
    fig_corr.update_layout(title="Multivariate Analysis - Correlation Matrix", height=600, width=800)

    save_plot(fig_corr, "multivariate_correlation.png")

import polars as pl
import json

def fix_json_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Detect and fix JSON-type columns in the Polars DataFrame."""
    print_section_header("Checking and fixing json types of columns")

    print("Detecting and fixing json types of columns")
    new_columns = []

    for col in df.columns:
        if df[col].dtype == pl.Utf8:  # Ensure column is a string type
            try:
                # Check if at least one non-null row is valid JSON
                sample_value = df[col].drop_nulls().filter(
                    df[col].drop_nulls().str.starts_with("{") & df[col].drop_nulls().str.ends_with("}")
                ).head(1)

                if len(sample_value) > 0:
                    # Convert the column into a struct by parsing JSON
                    df = df.with_columns(
                        pl.col(col).map_elements(lambda x: json.loads(x) if x else None).alias(col)
                    )

                    # Expand struct into separate columns
                    expanded_cols = df[col].struct.unnest().rename({k: f"{col}_{k}" for k in df[col].struct.fields})

                    # Drop original JSON column and merge expanded data
                    df = df.drop(col).hstack(expanded_cols)
                    print(f"✅ Fixed JSON column: {col}")

            except Exception as e:
                print(f"⚠️ Error processing column {col}: {e}")

    return df


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
    univariate_analysis(df)
    bivariate_analysis(df)
    multivariate_analysis(df)
    df=fix_json_columns(df)


    df=save_cleaned_data(df)
    return df 

