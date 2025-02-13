# Description: Custom functions for data preprocessing and feature engineering.

# Import necessary libraries
import pandas as pd
from typing import List, Union


# Function to convert columns to categorical variables
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

    # Return the DataFrame with converted column types
    return df


def main():
    print("This file contains custom functions for data preprocessing and feature engineering.")


if __name__ == "__main__":
    main()