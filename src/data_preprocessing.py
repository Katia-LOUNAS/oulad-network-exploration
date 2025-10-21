import pandas as pd
from pathlib import Path

def Load_ould_data(file_name: str) -> pd.DataFrame:
    """
    Load the OULAD dataset from a CSV file.

    Parameters:
    file_name (str): The name of the CSV file to load.

    Returns:
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    data_path = Path(__file__).parent / 'data' / file_name
    df = pd.read_csv(data_path)
    return df

def preprocess_ould_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the OULAD dataset by handling missing values and encoding categorical variables.

    Parameters:
    df (pd.DataFrame): The raw dataset.

    Returns:
    pd.DataFrame: The preprocessed dataset.
    """
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df