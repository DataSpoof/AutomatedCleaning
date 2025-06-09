# AutomatedCleaning
[![PyPI](https://img.shields.io/badge/PyPI-3775A9?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/AutomatedCleaning/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abhishek-kumar-singh-8a6326148/)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/dataspoof)
[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/abhi007si)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@dataspoof1977)


AutomatedCleaning is a Python library for automated data cleaning.It helps preprocess and analyze datasets by handling missing values, outliers, spelling corrections, and more.

![Logo](images/logo2.png)

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
- Detect and mask PII types of columns
- Performs univariate, bivariate, and multivariate analysis and save in a dashboard


## Installation
```bash
python3.11 -m venv .venv
.venv\Scripts\activate
uv pip install AutomatedCleaning==0.1.8
```

## Usage
It requires Claude API key which you can get it from here https://console.anthropic.com/settings/keys (Optional Step)

```bash
import automatedcleaning as ac

df = ac.load_data("Company_Data.csv")
df = ac.clean_data(df, background_image_path="assets/gradient.png")
```

## 🎥 Watch the Demo

[![Watch the video](https://i9.ytimg.com/vi_webp/yA5Tdq22_Yg/mqdefault.webp?v=68007439&sqp=CLCcmcIG&rs=AOn4CLA3PwRJEncEyiuIRTi1wMyyHkbY1g)](https://www.youtube.com/watch?v=yA5Tdq22_Yg)


