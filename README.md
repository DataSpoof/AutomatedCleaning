# AutomatedCleaning

AutomatedCleaning is a Python library for automated data cleaning.It helps preprocess and analyze datasets by handling missing values, outliers, spelling corrections, and more.

## Features
- Supports both large (100+ GB) and small datasets
- Detects and handles missing values and duplicate records
- Identifies and corrects spelling errors in categorical values
- Detect outliers
- Detects and fixes data imbalance
- Identifies and corrects skewness in numerical data
- Checks for correlation and detects multicollinearity
- Analyzes cardinality in categorical columns
- Identifies and cleans text columns
- Detect JSON-type columns
- Performs univariate, bivariate, and multivariate analysis


## Installation
```bash
pip install AutomatedCleaning
```

## Usage
```bash
import AutomatedCleaning as ac
df = ac.load_data("dataset.csv")
df_cleaned = ac.clean_data(df)
```
