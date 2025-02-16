# Description: Custom functions for data preprocessing and feature engineering.

# Import necessary libraries
import pandas as pd
import numpy as np
from typing import List, Union, Tuple
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt


# Convert string columns to categorical variables
def convert_to_categorical(
    df: pd.DataFrame, columns: Union[List[str], str] = "all"
) -> pd.DataFrame:
    """
    Convert specified columns of type object to categorical variables.
    If 'all' is given as columns, convert all string columns to categorical.

    Parameters:
    df (pd.DataFrame): The DataFrame to be processed.
    columns (list or str): List of columns to convert. Choosing 'all' will affect all string columns.

    Returns:
    pd.DataFrame: The DataFrame with converted columns.
    """
    if columns == "all":
        # Convert all object type columns to categorical
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype("category")
    elif isinstance(columns, list):
        # Convert specified columns to categorical
        for col in columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("category")
    else:
        raise ValueError(
            "Parameter 'columns' should be a list of column names or 'all'."
        )
    # Convert 'unknown' to None
    df = df.replace({"unknown": None})
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
    for col in df.select_dtypes(include=["category", "object"]).columns:
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
def probabilistic_imputation(
    df: pd.DataFrame, columns: Union[List[str], str] = "all"
) -> pd.DataFrame:
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

    if columns == "all":
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
                    np.random.choice(
                        range(len(values)), size=num_missing, p=probabilities
                    ),
                    categories=df[col].cat.categories,
                )
            else:
                df_imputed.loc[missing_indices, col] = np.random.choice(
                    values, size=num_missing, p=probabilities
                )

    return df_imputed


# Function to calculate the correlation between categorical columns and target variable
def show_categorical_correlation(df: pd.DataFrame, 
                                 target: str = 'y', 
                                 cols_to_consider: list = None, 
                                 show_details: bool = False, 
                                 figsize: tuple = (15, 10), 
                                 subplot_grid_size: Tuple[int, int] = None,
                                 annotate: bool = True) -> pd.DataFrame:
    
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
        subplot_grid_size: Tuple specifying the number of rows and columns for the subplot grid.
        annotate: If True, annotate the bars with percentage values.
    Returns:
        pd.DataFrame: A DataFrame containing the correlation analysis results.
    """

    if cols_to_consider is None:
        cols_to_consider = df.select_dtypes(include='object').columns.drop(target, errors='ignore').tolist()
    
    # Order the 'month' column if it is in the columns to consider
    if 'month' in cols_to_consider:
        month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)

    # Define the number of rows and columns for the subplot grid  
    # If the number of columns is not specified, calculate the optimal grid size
    if subplot_grid_size is None:  
        num_cols = len(cols_to_consider)
        ncols = int(np.ceil(np.sqrt(num_cols)))  # Number of columns in subplot grid
        nrows = int(np.ceil(num_cols / ncols))   # Number of rows in subplot grid

    else:
        nrows, ncols = subplot_grid_size[0], subplot_grid_size[1]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()  # Flatten the axes array for easier indexing
    
    # Create a DataFrame to store the results of the correlation analysis 
    df_cat_corr = pd.DataFrame(columns=["Column", "Chi2", "Cramer's V", "P-Value", "alpha", "is_significant"])    

    # Iterate over the columns to consider
    for idx, col in enumerate(cols_to_consider):
        contingency_table = pd.crosstab(df[col], df[target])

        # Chi-square test and Cramer's V
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        chi2, p = chi2.round(2), p.round(2)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt((chi2 / n) * ((min(contingency_table.shape) - 1)**-1)).round(2)
        alpha = 0.05  # Significance level  
        # Add results to the DataFrame
        row = pd.DataFrame([{"Column": col, "Chi2": chi2, "Cramer's V": cramers_v, "P-Value": p, "alpha": alpha, "is_significant": p < alpha}])
        df_cat_corr = pd.concat([df_cat_corr, row], ignore_index=True)   
       
        # Visualization (Stacked bar chart of proportions)
        proportions = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
        ax = proportions.plot(kind='bar', stacked=True, ax=axes[idx])
        axes[idx].set_title(f'Success rate by {col} (in % terms)')
        axes[idx].set_ylabel("Percentage share")

        if annotate is True:
            # Annotate bars with percentage values
            for p in ax.patches:
                width, height = p.get_width(), p.get_height()
                x, y = p.get_xy() 
                ax.annotate(f'{height:.0f}%', (x + width / 2, y + height / 2), ha='center', va='center')
    
    # If specified, return the DataFrame with correlation results      
    if show_details is True:            
        return df_cat_corr
    # Show all subplots
    plt.tight_layout()  # Adjust subplot params for a tight layout
    plt.show()  # Display the plot


# Function to plot the distribution of a column segmented by the target variable
def plot_relative_freq(df, col, target='y', figsize=(12, 3)):
    """
    Plots the relative frequency of each column segmented by the target variable.

    Args:
        df: The Pandas DataFrame.
        col: Column name.
        target: The name of the target variable.
        figsize: Tuple specifying the figure size for the plot.

    Returns:
        Two side-by-side plots showing the relative frequency of the column segmented by the target variable.
    """
    plt.figure(figsize=figsize)  # Adjust figure size for side-by-side plots

    # --- Left Plot: 
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    poutcome_counts = df.groupby([col, target]).size().unstack()    # Calculate total nums for each value and 'y' combination
    poutcome_counts.plot(kind='bar', stacked=False, color=['red', 'green'], ax=plt.gca())   # Plot the counts as a stacked bar chart
    # Add labels and title
    plt.title(f'Distr. of {col}* segmented by outcome')
    plt.ylabel('Number of occurrences')
    plt.xticks(rotation=0)  # Rotate x-axis labels if needed

    # --- Right Plot:
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    poutcome_proportions = poutcome_counts.div(poutcome_counts.sum(axis=1), axis=0)  # Calculate proportions of 'y' within 'column'
    poutcome_proportions.plot(kind='bar', stacked=True, ax=plt.gca())   # Plot the proportions as a stacked bar chart
    # Add labels and title
    plt.title(f'Deposit rate by {col}*')
    plt.ylabel('Proportions')
    plt.xticks(rotation=0)  # Rotate x-axis labels if needed
    # Add proportion labels on top of bars
    for p in plt.gca().patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        plt.gca().annotate(f'{height:.2f}', (x + width / 2, y + height / 2), ha='center', va='center', fontsize=8, color='black')
    plt.tight_layout()  # Adjust the plots to fit into the figure area
    plt.show()  # Display the plots


# Function to plot relationship between given columns and the target variable via boxplots
def plot_boxplots(
    df: pd.DataFrame,
    columns: List[str], 
    target: str = 'y', 
    figsize: Tuple[int, int] = (15, 3), 
    subplot_grid_size: Tuple[int, int] = None
):

    """
    Plots boxplots between given columns and the target variable.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (List[str]): List of columns to plot.
    target (str): The name of the target variable column.
    figsize (Tuple[int, int]): Figure size for the plots.
    subplot_grid_size (Tuple[int, int]): Grid size for the subplots (rows, columns).

    Returns:
    None
    """
    if subplot_grid_size is None:
        # Define the number of rows and columns for the subplot grid
        ncols = 2
        nrows = int(np.ceil(len(columns) / ncols))
    else:
        nrows, ncols = subplot_grid_size

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    # Iterate over the specified columns and create a boxplot for each one
    for idx, col in enumerate(columns):
        sns.boxplot(
            data=df, x=col, ax=axes[idx], hue=target, medianprops={"color": "red", "linewidth": 2}
        )
        axes[idx].set_title(f'Distribution by {col}*')
        axes[idx].grid(True)

    # Remove any unused subplots
    for idx in range(len(columns), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.grid(True)
    plt.show()


# if function is called directly, run the following code
if __name__ == "__main__":   
    pass
