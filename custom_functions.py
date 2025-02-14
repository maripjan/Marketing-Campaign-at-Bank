# Description: Custom functions for data preprocessing and feature engineering.

# Import necessary libraries
import pandas as pd
import numpy as np
from typing import List, Union, Tuple
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt


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


# Function to calculate the correlation between categorical columns and target variable
def show_categorical_correlation(df: pd.DataFrame, target: str = 'y', cols_to_consider: list = None, show_details: bool = False, figsize: tuple = (15, 10)):
    """
    Analyzes the relationship between a target variable and multiple 
    categorical columns, displaying results in a subplot grid.

    Args:
        df: The Pandas DataFrame.
        target: The name of the target variable column.
        cols_to_consider: A list of categorical column names. If None, uses all 
                          categorical columns except the target.
        show_details: If True, prints detailed results for each column.
        figsize: Tuple specifying the figure size for the plot grid.

    Returns:
        None (displays plots and prints results).
    """

    if cols_to_consider is None:
        cols_to_consider = df.select_dtypes(include='object').columns.drop(target, errors='ignore').tolist()

    num_cols = len(cols_to_consider)
    ncols = int(np.ceil(np.sqrt(num_cols)))  # Number of columns in subplot grid
    nrows = int(np.ceil(num_cols / ncols))   # Number of rows in subplot grid

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    for idx, col in enumerate(cols_to_consider):
        contingency_table = pd.crosstab(df[col], df[target])

        # Chi-square test and Cramer's V
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt((chi2 / n) * ((min(contingency_table.shape) - 1)**-1))       

        # If specified, show the proportions of 'yes' for each category        
        if show_details:
            print(f"--- Analysis for column: {col} ---")
            print(f"Chi-square statistic: {chi2:.2f} (P-value: {p:.2f})")        
            print(f"Cramer's V: {cramers_v:.2f}")
            # Show if the association is statistically significant
            alpha = 0.05
            if p < alpha:
                print(f"Statistically significant association (p < {alpha})")
            else:
                print(f"No statistically significant association (p >= {alpha})")
            # Calculate proportions of 'yes' for each category
            yes_proportions = contingency_table[df[target].unique()[1]] / contingency_table.sum(axis=1) if len(df[target].unique()) > 1 else contingency_table[df[target].unique()[0]] / contingency_table.sum(axis=1)
            print("\nProportion of 'yes' for each value in a category:")
            print(yes_proportions)
            print("-" * 50)  # Separator between columns

        # Visualization (Stacked bar chart of proportions)
        (contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100).plot(kind='bar', stacked=True, ax=axes[idx])
        axes[idx].set_title(f'Success rate by {col} (in % terms)')
        axes[idx].set_ylabel('Percentage share')       
    # Show all subplots
    plt.tight_layout()  # Adjust subplot params for a tight layout
    plt.show()


# if function is called directly, run the following code
if __name__ == "__main__":
    print("This file contains custom functions for data preprocessing and feature engineering.")
    pass
