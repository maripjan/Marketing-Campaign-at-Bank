import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import json
from typing import List, Union


class CleanImportedData(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols_to_drop = ['month', 'default', 'housing', 'loan']
        X = X.drop(columns=cols_to_drop)  # Drop initially useless columns
        X = X.replace({'unknown': None})  # Replace original values for certain columns
        X = probabilistic_imputation(X)  # Impute N/A using relative frequency method
        X = convert_to_categorical(X, columns='all')  # Convert all string columns to categorical
        X['duration_mins'] = (X['duration'] / 60).round().astype(int)  # Express call duration in mins and round to nearest integer
        X = X.drop(columns=['duration'], errors='ignore')  # Drop the original column where duration was calculated in seconds
        X['education'] = X['education'].replace('illiterate', 'basic.4y')  # Replace extremely rare value "illiterate" with the closest value "basic.4years"
        return X


class RegroupCategories(BaseEstimator, TransformerMixin):
    def __init__(self, mappings='mappings.json'):
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        with open(self.mappings, 'r') as f:
            mappings = json.load(f)
        # Map categrories from file
        education_level_mapping = mappings['education_level_mapping']
        job_type_mapping = mappings['job_category_mapping']
        income_level_mapping = mappings['income_level_mapping']

        X['education_level'] = X['education'].map(education_level_mapping).astype('category')  # Map education levels
        X['job_type'] = X['job'].map(job_type_mapping).astype('category')  # Map job categories
        X['income_level'] = X['job'].map(income_level_mapping).astype('category')  # Map income levels

        # Create new boolean columns based on previous contact attempts and deposit outcomes
        X['was_contacted_before'] = X['previous'].apply(lambda previous: True if previous > 0 else False)
        X['deposited_before'] = X['poutcome'].apply(lambda outcome: True if outcome == 'success' else False) 
        return X


class TruncateValues(BaseEstimator, TransformerMixin):
    def __init__(self, mappings='mappings.json'):
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        with open(self.mappings, 'r') as file:  # Load truncation values from JSON file
            mappings = json.load(file)
        truncation_values = mappings['truncation_values']
        for col, limits in truncation_values.items():
            X = truncate_values(X, col, lower=limits['lower'], upper=limits['upper'])
        return X


class OneHotEncodeCategorical(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, drop='first')

    def fit(self, X, y=None):
        categorical_columns = X.select_dtypes(include=['category', 'object']).columns
        self.encoder.fit(X[categorical_columns])
        return self

    def transform(self, X):
        categorical_columns = X.select_dtypes(include=['category', 'object']).columns
        encoded_columns = self.encoder.transform(X[categorical_columns])
        encoded_df = pd.DataFrame(encoded_columns, columns=self.encoder.get_feature_names_out(categorical_columns))
        X = X.drop(columns=categorical_columns)
        X = pd.concat([X.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        return X


class ScaleNumericalColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        self.scaler.fit(X[numeric_columns])
        return self

    def transform(self, X):
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X[numeric_columns] = self.scaler.transform(X[numeric_columns])
        return X


# Define the functions used in the transformers
def probabilistic_imputation(df: pd.DataFrame, columns: Union[List[str], str] = "all") -> pd.DataFrame:
    df_imputed = df.copy()  # Create a copy to avoid modifying the original
    if columns == "all":
        columns = df.columns  # Impute all columns if none specified
    for col in columns:
        if df[col].isnull().any():  # Check if there are NaNs in the column
            value_counts = df[col].value_counts(normalize=True)
            values = value_counts.index
            probabilities = value_counts.values
            missing_indices = df[col].isnull()
            num_missing = missing_indices.sum()
            if pd.api.types.is_categorical_dtype(df[col]):
                df_imputed.loc[missing_indices, col] = pd.Categorical.from_codes(
                    np.random.choice(range(len(values)), size=num_missing, p=probabilities),
                    categories=df[col].cat.categories,
                )
            else:
                df_imputed.loc[missing_indices, col] = np.random.choice(values, size=num_missing, p=probabilities)
    return df_imputed


def convert_to_categorical(df: pd.DataFrame, columns: Union[List[str], str] = "all") -> pd.DataFrame:
    if columns == 'all':
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
    elif isinstance(columns, list):
        for col in columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
    else:
        raise ValueError("Parameter 'columns' should be a list of column names or 'all'.")
    df = df.replace({'unknown': None})
    return df


def truncate_values(df: pd.DataFrame, column: str, lower=None, upper=None) -> pd.DataFrame:
    if lower is None:
        lower = df[column].quantile(0.05)
    if upper is None:
        upper = df[column].quantile(0.95)
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df
