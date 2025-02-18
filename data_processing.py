import pandas as pd
import sqlite3
import custom_functions as cf
import json


# Class to read input data from different formats
class Input:
    def __init__(self, df_path):        
        self.df_path = df_path      
    
    # Method to clean data immediately after reading
    def clean_imported_data(self):
        cols_to_drop = ['month', 'default', 'housing', 'loan']       
        self.df = self.df.drop(columns=cols_to_drop)  # Drop initially useless columns
        self.df = self.df.replace({'unknown': None})  # Replace original values for certain columns
        self.df = cf.probabilistic_imputation(self.df)  # Impute N/A using relative frequency method
        self.df = cf.convert_to_categorical(self.df, columns='all')  # Convert all string columns to categorical
        self.df['duration_mins'] = (self.df['duration'] / 60).round().astype(int)  # Express call duration in mins and round to nearest integer
        
        # Drop the original column where duration was calculated in seconds
        self.df = self.df.drop(columns=['duration'], errors='ignore')

        # Replace extermely rare value "illeterate" with the closest value "basic.4years"
        self.df['education'] = self.df['education'].replace('illiterate', 'basic.4y')
        return self.df

    # Method to regroup categories in the data according to existing mappings
    def regroup_categories(self, mappings='mappings.json'):       
        # Load mappings from JSON file
        with open(mappings, 'r') as f:
            mappings = json.load(f)
        
        education_level_mapping = mappings['education_level_mapping']
        job_type_mapping = mappings['job_category_mapping']
        income_level_mapping = mappings['income_level_mapping']

        self.df['education_level'] = self.df['education'].map(education_level_mapping).astype('category')  # Map education levels
        self.df['job_type'] = self.df['job'].map(job_type_mapping).astype('category')  # Map job categories
        self.df['income_level'] = self.df['job'].map(income_level_mapping).astype('category')  # Map income levels

        # Create new boolean columns based on previous contact attempts and deposit outcomes
        self.df['was_contacted_before'] = self.df['previous'].apply(lambda previous: True if previous > 0 else False)
        self.df['deposited_before'] = self.df['poutcome'].apply(lambda outcome: True if outcome == 'success' else False)
       
        return self.df

    # Function that does end-to-end preprocessing    
    def preprocess(self, mappings='mappings.json'):
        self.df = pd.read_csv(self.df_path)  
        self.df = self.clean_imported_data()
        self.df = self.regroup_categories()
        
        # Truncate numerical cols to exclude outliers using values from mappings        
        with open(mappings, 'r') as file:   # Load truncation values from JSON file
            mappings = json.load(file)        
        truncation_values = mappings['truncation_values']       
        for col, limits in truncation_values.items():
            self.df = cf.truncate_values(self.df, col, lower=limits['lower'], upper=limits['upper'])   
        # Finally, remove all duplicates
        self.df = self.df.drop_duplicates()     
        return self.df


# Class to save output to different formats
class Output:
    def __init__(self, df):
        self.df = df
    
    def to_sqlite(self, db_name, table_name):
        with sqlite3.connect(db_name) as conn:
            self.df.to_sql(table_name, conn, if_exists='append', index=False)        
    
    def to_csv(self, file_name):
        self.df.to_csv(file_name, index=False)
    
    def to_parquet(self, file_name):
        self.df.to_parquet(file_name, index=False)


if __name__ == '__main__':
    pass
# Example usage:
# input_processor = Input('data.csv')
# input_processor.process_data()
# output_processor = Output(input_processor.df)
# output_processor.to_sqlite('database.db', 'table_name')
# output_processor.to_csv('output.csv')
# output_processor.to_parquet('output.parquet')