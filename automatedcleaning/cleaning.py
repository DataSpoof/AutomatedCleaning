import polars as pl
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import math
import os
import missingno as msno
import logging
import getpass
from langchain_anthropic import ChatAnthropic
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress all logging from Presidio and other libraries
logging.getLogger().setLevel(logging.ERROR)

# Suppress NLTK download messages
nltk_logger = logging.getLogger('nltk')
nltk_logger.setLevel(logging.ERROR)

# OR Suppress all warnings in NLTK
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


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
nltk.download('punkt_tab',quiet=True)
nltk.download('punkt',quiet=True)  # For tokenization
nltk.download('stopwords',quiet=True)  # For stopword removal
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



def replace_symbols(df):
    """Replace symbols and convert to float using Polars."""
    
    # Identify columns containing unwanted symbols
    problematic_cols = [
        col for col in df.columns 
        if df[col].cast(pl.Utf8, strict=False).str.contains(r'[\$,₹,-]', literal=False).any()
    ]
    
    
    # Replace symbols and convert to float
    df = df.with_columns([
        pl.col(col)
        .str.replace_all(r'[\$,₹,-,•,-,-]', '')  # Remove $, ₹, - symbols
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

# Initialize logging
logging.basicConfig(filename="corrections.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# System prompt for categorical value correction
CATEGORICAL_CORRECTION_PROMPT = """
You are a data validation expert. Your task is to correct spelling errors and inconsistencies in categorical values.
- Ensure all values are standardized and correctly spelled.
- Return the corrected values **in the same order** as provided.
- Use a bullet-point format (one value per line).

Example input:
Column: ProductCategory
Values: ['elecronics', 'fashon', 'Electronics', 'fasihon', 'home_appl']

Expected output:
- Electronics
- Fashion
- Electronics
- Fashion
- Home Appliances

Example input:
Column: ProductCategory
Values: ['good', 'bad', 'goo', 'ba']

Expected output:
- good
- bad
- good
- bad

"""



def fix_spelling_errors_in_categorical(df):
    """Fix spelling errors in categorical columns using Claude AI or manual input."""
    print_section_header("Fix spelling errors in categorical columns")
    
    user_choice = input("Do you want to correct spelling errors in categorical columns? (yes/no): ").strip().lower()
    if user_choice not in ["yes", "y"]:
        print("Skipping spell-checking.")
        return df

    method_choice = input("Choose correction method: (1) Automatic (Claude AI) (2) Manual: ").strip()
    
    if method_choice == "1":
        api_key = getpass.getpass("Enter your Claude API key: ")
        model = ChatAnthropic(model="claude-3-7-sonnet-latest", api_key=api_key)
    else:
        model = None  # No AI model needed for manual correction

    categorical_columns = [col for col in df.columns if df[col].dtype == pl.Utf8]

    for col in categorical_columns:
        unique_values = df[col].drop_nulls().unique().to_list()
        
        if method_choice == "1":
            corrected_values = correct_categorical_values(model, col, unique_values)
        else:
            corrected_values = manual_correction(col, unique_values)
        
        correction_map = dict(zip(unique_values, corrected_values))
        
        print(f"\nColumn: {col}\nOriginal Values: {unique_values}\nCorrected Values: {corrected_values}")

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

    # Ensure response is properly formatted and split into a clean list
    corrected_values = response.content.strip().split("\n")

    # If response length is incorrect, return original values
    if len(corrected_values) != len(values):
        logging.warning(f"Unexpected response format for column '{column_name}'. Received: {response.content}")
        return values  # Return original values if there's an issue

    return corrected_values


def manual_correction(column_name: str, values: list) -> list:
    """Allows user to manually correct categorical values."""
    corrected_values = []
    print(f"\nManual correction for column: {column_name}")
    
    for val in values:
        new_val = input(f"Correct '{val}' (press enter to keep it unchanged): ").strip()
        corrected_values.append(new_val if new_val else val)
    
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
import numpy as np

def check_problem_type(df, target_col):
    """Determine whether the problem is classification or regression."""
    unique_values = df[target_col].n_unique()
    
    if df[target_col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64] and unique_values > 10:
        print(f"Detected a regression problem (continuous target column '{target_col}'). Skipping class balancing.")
        return "regression"
    else:
        print(f"Detected a classification problem (categorical/discrete target column '{target_col}').")
        return "classification"

def check_and_handle_imbalance(df, target_col):
    """
    Check for class imbalance and handle it using user-selected undersampling or oversampling.
    
    Parameters:
    - df (pl.DataFrame): The input Polars dataframe
    - target_col (str): The name of the target column
    
    Returns:
    - pl.DataFrame: A balanced dataframe or the original if skipped
    """
    
    if check_problem_type(df, target_col) == "regression":
        return df  # Skip class balancing if it's a regression problem
    
    # Original Class Distribution
    class_counts = df[target_col].value_counts().sort("count")
    min_count, max_count = class_counts["count"].min(), class_counts["count"].max()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print("Original Class Distribution:")
    print(class_counts)

    if imbalance_ratio > 1.5:
        print(f"\nThe target column '{target_col}' is **imbalanced**.")
        
        # Ask the user for input
        method = input("\nChoose a balancing method - 'oversampling', 'undersampling', or press Enter to skip: ").strip().lower()
        
        if method == "":
            print("\nSkipping class balancing.")
            return df

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
            raise ValueError("\nInvalid method. Please choose 'oversampling', 'undersampling', or press Enter to skip.")

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

import os
import math
import base64
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import pandas as pd
from plotly.subplots import make_subplots

def save_dashboard(html_content, filename="dashboard.html"):
    OUTPUT_DIR = "output/eda/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
        f.write(html_content)

def generate_dashboard(df: pl.DataFrame, background_image_path: str):
    df_pandas = df.to_pandas()
    num_cols = df.select(pl.col(pl.Float64, pl.Int64)).columns
    cat_cols = df.select(pl.col(pl.Utf8)).columns

    figures_univariate, figures_bivariate, figures_multivariate = [], [], []

    kpi_html = '<div class="kpi-container">'
    for col in num_cols[:4]:
        total = df[col].sum()
        kpi_html += f'''
        <div class="kpi-box">
            <h3>{round(total, 2):,}</h3>
            <p>Total {col}</p>
        </div>
        '''
    kpi_html += '</div>'

    for col in cat_cols:
        value_counts = df_pandas[col].value_counts().reset_index()
        value_counts.columns = [col, "count"]
        fig = px.bar(value_counts, x=col, y="count", title=f"{col} Distribution", color_discrete_sequence=["#00BFFF"])
        fig.update_layout(template='plotly_dark', bargap=0.4)
        figures_univariate.append(fig)

    for col in num_cols:
        fig1 = px.histogram(df_pandas, x=col, title=f"Histogram of {col}", color_discrete_sequence=["#FFA500"])
        fig1.update_layout(template='plotly_dark', bargap=0.4)
        figures_univariate.append(fig1)

        fig2 = px.box(df_pandas, y=col, title=f"Boxplot of {col}", color_discrete_sequence=["#7CFC00"])
        fig2.update_layout(template='plotly_dark')
        figures_univariate.append(fig2)

    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            fig = px.scatter(df_pandas, x=num_cols[i], y=num_cols[j], title=f"{num_cols[i]} vs {num_cols[j]}", color_discrete_sequence=["#FF69B4"])
            fig.update_layout(template='plotly_dark')
            figures_bivariate.append(fig)

    for cat in cat_cols:
        for num in num_cols:
            fig = px.histogram(df_pandas, x=num, color=cat, barmode='stack', title=f"{num} by {cat}")
            fig.update_layout(template='plotly_dark', bargap=0.4)
            figures_bivariate.append(fig)

    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            grouped_data = df_pandas.groupby([cat_cols[i], cat_cols[j]]).size().reset_index(name='count')
            fig = px.bar(grouped_data, x=cat_cols[i], y='count', color=cat_cols[j], barmode='stack',
                         title=f"{cat_cols[i]} vs {cat_cols[j]}")
            fig.update_layout(template='plotly_dark', bargap=0.4)
            figures_bivariate.append(fig)

    if len(num_cols) > 1:
        corr_matrix = df_pandas[num_cols].corr()
        fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns,
                                        y=corr_matrix.index, colorscale='Blues', zmin=-1, zmax=1))
        fig.update_layout(title="Correlation Heatmap", template='plotly_dark')
        figures_multivariate.append(fig)

    with open(background_image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()


    def create_subplot(figures, title):
        if not figures:
            return "<p>No plots available.</p>"

        cols = 3  # Fixed to 3 plots per row
        rows = math.ceil(len(figures) / cols)
        safe_spacing = min(0.05, 1 / (rows - 1)) if rows > 1 else 0.05

        subplot_fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=[fig.layout.title.text for fig in figures],
            vertical_spacing=safe_spacing
        )

        for i, fig in enumerate(figures):
            for trace in fig.data:
                row = (i // cols) + 1
                col = (i % cols) + 1
                subplot_fig.add_trace(trace, row=row, col=col)

        # Dynamic height: 400px per row (you can adjust 400 to your preference)
        height = 400 * rows

        subplot_fig.update_layout(
            title_text=title, 
            height=height, 
            width=1500, 
            showlegend=False, 
            template='plotly_dark'
        )
        return subplot_fig.to_html(full_html=False)


    def create_bivariate_subplot(figures, title):
        if not figures:
            return "<p>No plots available.</p>"

        cols = 3
        rows = math.ceil(len(figures) / cols)

        # Reduce vertical spacing between rows
        vertical_spacing = 0.03  # try 0.03 or even 0.02

        subplot_fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[fig.layout.title.text for fig in figures],
            vertical_spacing=vertical_spacing,
            horizontal_spacing=0.05
        )

        for i, fig in enumerate(figures):
            row = (i // cols) + 1
            col = (i % cols) + 1
            for trace in fig.data:
                subplot_fig.add_trace(trace, row=row, col=col)

        base_height_per_row = 400  # reasonable height per row
        extra_height_padding = 100  # space for title/margin

        height = base_height_per_row * rows + extra_height_padding

        subplot_fig.update_layout(
            title_text=title,
            height=height,
            width=1500,
            showlegend=False,
            template='plotly_dark',
            margin=dict(l=40, r=40, t=80, b=40)
        )

        return subplot_fig.to_html(full_html=False)


    def create_single_plot(figures, title):
        if not figures:
            return "<p>No plots available.</p>"

        # Take only the first figure, assuming only one for bivariate/multivariate
        fig = figures[0]

        fig.update_layout(
            title_text=title,
            height=600,
            width=800,
            showlegend=True,
            template='plotly_dark',
            margin=dict(l=80, r=40, t=80, b=40)
        )

        return fig.to_html(full_html=False)


    html_content = f"""
    <html>
    <head>
        <title>PowerBI Styled EDA Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Segoe UI, sans-serif;
                margin: 0;
                padding: 0;
                background-image: url("data:image/png;base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                color: white;
            }}
            h1 {{
                color: #FFFFFF;
                text-align: center;
                padding-top: 20px;
                text-shadow: 1px 1px 2px #000;
            }}
            .navbar {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                background-color: rgba(0,0,0,0.7);
                padding: 10px 0;
                position: sticky;
                top: 0;
                z-index: 1000;
            }}
            .navbar button {{
                background-color: #00BFFF;
                border: none;
                color: white;
                padding: 10px 20px;
                margin: 5px;
                cursor: pointer;
                font-size: 16px;
                border-radius: 5px;
                transition: background-color 0.3s;
            }}
            .navbar button:hover {{
                background-color: #009ACD;
            }}
            .section {{
                display: none;
                padding: 20px;
            }}
            .active {{
                display: block;
            }}
            .kpi-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 10px;
                padding: 0 20px;
            }}
            .kpi-box {{
                background-color: rgba(0,0,0,0.6);
                padding: 20px;
                border-radius: 10px;
                width: 200px;
                text-align: center;
                box-shadow: 0 0 10px #333;
            }}
            .kpi-box h3 {{
                font-size: 28px;
                margin: 0;
                color: #00FFFF;
            }}
            .kpi-box p {{
                font-size: 16px;
                margin: 5px 0 0 0;
                color: #CCCCCC;
            }}
            .user-input {{
                text-align: center;
                padding: 20px;
            }}
            .user-input input {{
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
                width: 250px;
                margin-right: 10px;
            }}
            .user-input button {{
                padding: 10px 20px;
                font-size: 16px;
                background-color: #00BFFF;
                border: none;
                border-radius: 5px;
                color: white;
                cursor: pointer;
            }}
            .user-charts {{
                margin-top: 20px;
            }}
        </style>
        <script>
            function showSection(sectionId) {{
                var sections = document.getElementsByClassName('section');
                for (var i = 0; i < sections.length; i++) {{
                    sections[i].classList.remove('active');
                }}
                document.getElementById(sectionId).classList.add('active');
            }}

            function generatePieChart(columnName) {{
                const data = JSON.parse(document.getElementById('data-json').textContent);
                if (!data.hasOwnProperty(columnName)) {{
                    alert('Invalid column name!');
                    return;
                }}
                const values = data[columnName];
                const counts = values.reduce((acc, val) => {{
                    acc[val] = (acc[val] || 0) + 1;
                    return acc;
                }}, {{}});

                const labels = Object.keys(counts);
                const vals = Object.values(counts);

                const pieData = [{{
                    type: 'pie',
                    labels: labels,
                    values: vals
                }}];

                const layout = {{
                    title: 'Pie Chart of ' + columnName,
                    paper_bgcolor: 'rgba(0,0,0,0.5)',
                    font: {{ color: 'white' }}
                }};

                const divId = 'pie_' + columnName + '_' + Math.floor(Math.random() * 100000);
                const chartDiv = document.createElement('div');
                chartDiv.id = divId;
                chartDiv.style.marginTop = '30px';
                document.getElementById('user-charts').appendChild(chartDiv);
                Plotly.newPlot(divId, pieData, layout);
            }}

            window.onload = function() {{
                showSection('univariate');
            }};
        </script>
    </head>
    <body>
        <h1>Power BI Styled EDA Dashboard</h1>

        {kpi_html}

        <div class="navbar">
            <button onclick="showSection('univariate')">Univariate</button>
            <button onclick="showSection('bivariate')">Bivariate</button>
            <button onclick="showSection('multivariate')">Multivariate</button>
        </div>

        <div id="univariate" class="section">
            <h2>Univariate Analysis</h2>
            {create_subplot(figures_univariate, "Univariate Analysis")}
        </div>

        <div id="bivariate" class="section">
            <h2>Bivariate Analysis</h2>
            {create_bivariate_subplot(figures_bivariate, "Bivariate Analysis")}
        </div>

        <div id="multivariate" class="section">
            <h2>Multivariate Analysis</h2>
            {create_single_plot(figures_multivariate, "Multivariate Analysis")}
        </div>

        
       
    </body>
    </html>
    """

    save_dashboard(html_content)




import json

def fix_json_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Detect and fix JSON-type columns in the Polars DataFrame."""
    print_section_header("Checking and fixing json types of columns")

    print("Detecting and fixing json types of columns if there are any")
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


import polars as pl
from typing import Dict, List
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
import warnings
import logging

# Suppress logs and warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

def detect_and_mask_pii_polars(df: pl.DataFrame, sample_size: int = 10) -> pl.DataFrame:
    """
    Detects and masks PII in a Polars DataFrame.

    Args:
        df (pl.DataFrame): Input Polars DataFrame.
        sample_size (int): Number of rows to sample for PII detection.

    Returns:
        pl.DataFrame: DataFrame with masked PII columns added.
    """
    print_section_header("Checking for any PII types in columns and masking them")

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    pii_columns: Dict[str, List[str]] = {}

    # Step 1: Detect PII columns using sampling
    for col in df.columns:
        col_data = df[col].drop_nulls().cast(str)
        if len(col_data) == 0:
            continue
        sample = col_data.sample(min(sample_size, len(col_data)), seed=42)
        detected_types = set()

        for value in sample.to_list():
            results = analyzer.analyze(text=value, language='en')
            for r in results:
                detected_types.add(r.entity_type)

        if detected_types:
            pii_columns[col] = list(detected_types)

    # Print detected columns info or message if none found
    if pii_columns:
        print("Detected PII types in columns:")
        for col, types in pii_columns.items():
            print(f" - Column '{col}': {types}")
    else:
        print("No PII columns detected in the dataset.")

    # Step 2: Mask PII in detected columns
    df_dict = df.to_dict(as_series=False)  # Convert to dict for easy row-wise updates

    for col in pii_columns:
        masked_values = []
        for text in df_dict[col]:
            text = str(text) if text is not None else ""
            results = analyzer.analyze(text=text, language='en')
            if results:
                masked = anonymizer.anonymize(text=text, analyzer_results=results).text
                masked_values.append(masked)
            else:
                masked_values.append(text)

        df_dict[f"{col}_masked"] = masked_values

    return pl.DataFrame(df_dict)



def clean_data(df, background_image_path=None):
    """Main function to clean the data."""
    df = detect_column_types_and_process_text(df)
    df = handle_negative_values(df)
    df = replace_symbols_and_convert_to_float(df)
    df = fix_spelling_errors_in_columns(df)
    df = fix_spelling_errors_in_categorical(df)
    df = replace_symbols(df)

    df = handle_missing_values(df)
    df = handle_duplicates(df)
    check_cardinality(df)

    # df = remove_outliers(df)
    df = fix_skewness(df)
    df = check_multicollinearity(df)

    print_section_header("Enter target column (or press Enter to skip)")
    target_col = input("Enter the target column: ").strip()
    
    if target_col:
        if target_col in df.columns:
            df = check_and_handle_imbalance(df, target_col)
        else:
            print(f"⚠️ Warning: '{target_col}' not found in columns. Skipping imbalance handling.")
    else:
        print("ℹ️ Skipping target column–based imbalance check.")

    # Optional dashboard background
    if background_image_path:
        generate_dashboard(df, background_image_path=background_image_path)
    else:
        generate_dashboard(df)

    df = fix_json_columns(df)
    masked_df = detect_and_mask_pii_polars(df)
    df = save_cleaned_data(masked_df)
    return df
