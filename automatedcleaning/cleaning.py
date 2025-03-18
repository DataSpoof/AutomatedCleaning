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
        return pl.read_csv(file_path,infer_schema_length=10000,try_parse_dates=True,ignore_errors=False)
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



import json

contractions = {"ain't": "am not", "aren't": "are not", "can't": "cannot", 
"can't've": "cannot have", "'cause": "because", "could've": "could have", 
"couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", 
"doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
"hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have",
"he'll": "he will", "he's": "he is", "how'd": "how did", "how'll": "how will",
"how's": "how is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
"isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
"let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
"mightn't": "might not", "must've": "must have", "mustn't": "must not",
"needn't": "need not", "oughtn't": "ought not", "shan't": "shall not",
"sha'n't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is",
"should've": "should have", "shouldn't": "should not", "that'd": "that would",
"that's": "that is", "there'd": "there had", "there's": "there is", "they'd": "they would",
"they'll": "they will", "they're": "they are", "they've": "they have", "wasn't": "was not",
"we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
"weren't": "were not", "what'll": "what will", "what're": "what are", "what's": "what is",
"what've": "what have", "where'd": "where did", "where's": "where is", "who'll": "who will",
"who's": "who is", "won't": "will not", "wouldn't": "would not", "you'd": "you would",
"you'll": "you will", "you're": "you are", "wfh": "work from home", "wfo": "work from office",
"idk": "i do not know", "brb": "be right back", "btw": "by the way", "tbh": "to be honest",
"omw": "on my way", "lmk": "let me know", "fyi": "for your information",
"imo": "in my opinion", "smh": "shaking my head", "nvm": "never mind",
"ikr": "i know right", "fr": "for real", "rn": "right now", "gg": "good game",
"dm": "direct message", "afaik": "as far as i know", "bff": "best friends forever",
"ftw": "for the win", "hmu": "hit me up", "ggwp": "good game well played"}



import re
import nltk
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt_tab')
nltk.download('punkt')  # For tokenization
nltk.download('stopwords')  # For stopword removal
from nltk.stem import WordNetLemmatizer


def preprocess_text(text, remove_stopwords=True):
    if text is None or not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.split()
    new_text = [contractions[word] if word in contractions else word for word in text]
    text = " ".join(new_text)
    
    # Remove URLs, usernames, special characters, and emojis
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'"', ' ', text)
    
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF" 
                               "\U0001F680-\U0001F6FF\U0001F700-\U0001F77F" 
                               "\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF" 
                               "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F" 
                               "\U0001FA70-\U0001FAFF\U00002702-\U000027B0" 
                               "\U000024C2-\U0001F251]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Tokenization
    words = word_tokenize(text)
    
    # Stopword removal
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    

    return " ".join(words)




def detect_column_types_and_process_text(df):
    """
    Detect whether string columns in a Polars DataFrame are likely categorical, text, or JSON.
    
    This function categorizes string columns as:
    - 'categorical': Columns with relatively few unique values compared to total rows
    - 'text': Columns with many unique values and longer string lengths
    - 'json': Columns containing valid JSON objects/arrays
    
    Parameters:
    -----------
    df : pl.DataFrame
        The input Polars DataFrame to analyze
        
    Returns:
    --------
    tuple:
        (DataFrame with analysis results, set of categorical columns, set of text columns, set of json columns)
    """
    print_section_header("Checking whether column is string or text or json and peprocess text column")


    # Get only string columns
    string_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]
    
    if not string_cols:
        return (pl.DataFrame({"column": [], "type": [], "unique_ratio": [], "avg_length": []}), 
                set(), set(), set())
    
    results = []
    categorical_cols = set()
    text_cols = set()
    json_cols = set()
    
    # Number of rows in the dataframe
    row_count = df.height
    
    for col in string_cols:
        # Calculate metrics
        unique_count = df[col].n_unique()
        unique_ratio = unique_count / row_count if row_count > 0 else 0
        
        # Calculate average string length using a more reliable approach
        try:
            # Try different string length methods depending on Polars version
            avg_length = df.select(
                pl.mean(pl.col(col).cast(pl.Utf8).str.length()).alias("avg_length")
            ).item()
        except AttributeError:
            try:
                # Alternative approach using string_length expression
                avg_length = df.select(
                    pl.mean(pl.string_length(pl.col(col))).alias("avg_length")
                ).item()
            except:
                # Fallback to a manual calculation if needed
                non_null = df.filter(pl.col(col).is_not_null())
                if non_null.height > 0:
                    avg_length = sum(len(str(x)) for x in non_null[col].to_list()) / non_null.height
                else:
                    avg_length = 0
        
        # Check if column contains JSON
        is_json = False
        json_sample_count = min(30, df.height)  # Check up to 30 rows for efficiency
        
        if json_sample_count > 0:
            # Take a sample of non-null values
            sample_values = df.filter(pl.col(col).is_not_null()).head(json_sample_count)[col].to_list()
            
            # JSON detection heuristic: check if strings start with { or [ and parse as JSON
            json_count = 0
            for value in sample_values:
                value_str = str(value).strip()
                if (value_str.startswith('{') and value_str.endswith('}')) or \
                   (value_str.startswith('[') and value_str.endswith(']')):
                    try:
                        json.loads(value_str)
                        json_count += 1
                    except:
                        pass
            
            # If more than 50% of samples (lowered threshold for your dataset) are valid JSON, classify as JSON
            is_json = json_count > 0 and (json_count / len(sample_values) >= 0.5)
        
        # Determine column type
        if is_json:
            col_type = "json"
            json_cols.add(col)
        elif unique_ratio < 0.2 or (unique_count < 50 and avg_length < 20):
            col_type = "categorical"
            categorical_cols.add(col)
        else:
            col_type = "text"
            text_cols.add(col)
            pandas_series = df[col].to_pandas()
            processed_series = pandas_series.apply(preprocess_text)
            df = df.with_columns(pl.Series(col, processed_series))
 
            
        results.append({
            "column": col,
            "type": col_type,
            "unique_ratio": round(unique_ratio, 3),
            "avg_length": round(avg_length, 1) if avg_length is not None else None,
            "unique_count": unique_count,
            "is_json": is_json
        })
    
    result_df = pl.DataFrame(results)
    
    # Print the sets in the desired format
    print(f"Categorical columns are {categorical_cols}")
    print(f"Text columns are {text_cols}") 
    print(f"Json columns are {json_cols}")
    
    return df













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




import os
import json
import polars as pl
import logging
import getpass
from langchain_anthropic import ChatAnthropic

# Initialize logging
logging.basicConfig(filename="corrections.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# System prompt for categorical value correction
CATEGORICAL_CORRECTION_PROMPT = """
You are a data validation expert. Check if the given categorical values are valid.
Fix any incorrect values based on context and common categories.
Return only the corrected values.
"""

def fix_spelling_errors_in_categorical(df):
    """Fix spelling errors in categorical columns using Claude AI."""
    print_section_header("Fix spelling errors in categorical columns using Claude")

    api_key = getpass.getpass("Enter your Claude API key: ")
    model = ChatAnthropic(model="claude-3-7-sonnet-latest", api_key=api_key)
    
    categorical_columns = [col for col in df.columns if df[col].dtype == pl.Utf8]
    
    for col in categorical_columns:
        unique_values = df[col].drop_nulls().unique().to_list()
        corrected_values = correct_categorical_values(model, col, unique_values)
        correction_map = dict(zip(unique_values, corrected_values))
        print(f"The unique values are {unique_values} and the fixed values are {corrected_values} ")

        # Log corrections
        for old_value, new_value in correction_map.items():
            if old_value != new_value:
                logging.info(f"Column: {col} | '{old_value}' -> '{new_value}'")

        df = df.with_columns(pl.col(col).replace(correction_map).alias(col))
    
    return df

def correct_categorical_values(model, column_name: str, values: list) -> list:
    """Corrects categorical column values using Claude AI."""
    text = f"Column: {column_name}\nValues: {', '.join(values)}"
    model_input = [
        {"role": "system", "content": CATEGORICAL_CORRECTION_PROMPT},
        {"role": "user", "content": text},
    ]
    response = model.invoke(model_input)
    corrected_values = response.content.split(', ')  # Assuming response returns comma-separated values
    return corrected_values




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
    print_section_header("Checking for missing values and fixing it")
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
    print_section_header("Checking for duplicate values and fixing it")
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
    df=detect_column_types_and_process_text(df)
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

