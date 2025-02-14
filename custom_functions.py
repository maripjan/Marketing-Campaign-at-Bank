# Description: Custom functions for data preprocessing and feature engineering.

# Import necessary libraries
import pandas as pd
import numpy as np
from typing import List, Union, Tuple
from sklearn.preprocessing import LabelEncoder


# Convert string columns to categorical variables
def convert_to_categorical(df: pd.DataFrame, columns: Union[List[str], str] = 'all') -> pd.DataFrame:
    """
    Convert specified columns of type object to categorical variables.
    If 'all' is given as columns, convert all string columns to categorical.

    Parameters:
    df (pd.DataFrame): The DataFrame to be processed.
    columns (list or str): List of columns to convert. Choosing 'all' will affect all string columns.

    Returns:
    pd.DataFrame: The DataFrame with converted columns.
    """
    if columns == 'all':
        # Convert all object type columns to categorical
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')
    elif isinstance(columns, list):
        # Convert specified columns to categorical
        for col in columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
    else:
        raise ValueError("Parameter 'columns' should be a list of column names or 'all'.")
    # Convert 'unknown' to None
    df = df.replace({'unknown': None})
    # Return the DataFrame with converted column types
    return df


# Encode categorical columns to numeric values
def encode_categorical_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical columns to numeric values.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.

    Returns:
    Tuple[pd.DataFrame, dict]: A tuple containing the DataFrame with encoded categorical columns and a dictionary of LabelEncoders for each categorical column.
    """
    label_encoders = {}
    for col in df.select_dtypes(include=['category', 'object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders


# Decode numeric columns back to original categorical values
def decode_categorical_columns(df: pd.DataFrame, label_encoders: dict) -> pd.DataFrame:
    """
    Decode numeric columns back to original categorical values.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    label_encoders (dict): A dictionary of LabelEncoders for each categorical column.

    Returns:
    pd.DataFrame: The DataFrame with decoded categorical columns.
    """
    for col, le in label_encoders.items():
        df[col] = le.inverse_transform(df[col].astype(int))
    return df


# Impute missing values in specified columns with the most frequent value
def probabilistic_imputation(df: pd.DataFrame, columns: Union[List[str], str] = 'all') -> pd.DataFrame:
    """
    Imputes missing values in specified columns of a DataFrame
    by probabilistically assigning them based on the observed
    frequencies of non-missing values in each column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (list or str): A list of column names to impute. If 'all', imputes all columns.

    Returns:
    pd.DataFrame: A new DataFrame with imputed values.
    """
    df_imputed = df.copy()  # Create a copy to avoid modifying the original

    if columns == 'all':
        columns = df.columns  # Impute all columns if none specified

    for col in columns:
        if df[col].isnull().any():  # Check if there are NaNs in the column
            # Calculate the frequencies of non-missing values
            value_counts = df[col].value_counts(normalize=True)

            # Create a list of values and their probabilities
            values = value_counts.index
            probabilities = value_counts.values

            # Impute missing values probabilistically
            missing_indices = df[col].isnull()
            num_missing = missing_indices.sum()

            if pd.api.types.is_categorical_dtype(df[col]):
                df_imputed.loc[missing_indices, col] = pd.Categorical.from_codes(
                    np.random.choice(range(len(values)), size=num_missing, p=probabilities),
                    categories=df[col].cat.categories
                )
            else:
                df_imputed.loc[missing_indices, col] = np.random.choice(values, size=num_missing, p=probabilities)

    return df_imputed


# if function is called directly, run the following code
if __name__ == "__main__":
    print("This file contains custom functions for data preprocessing and feature engineering.")
    pass
