import pandas as pd
import sqlite3


# Class to read input data from different formats
class Input:
    def __init__(self, df_path):
        self.df = pd.read_csv(df_path)
    
    def process_data(self):
        # Example processing steps
        self.df.drop(columns=['unnecessary_column'], inplace=True)
        self.df['date_column'] = pd.to_datetime(self.df['date_column'])
        self.df['numeric_column'] = self.df['numeric_column'].replace({-1: 0})
        self.df['new_column'] = self.df['existing_column'] * 2


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

# Example usage:
# input_processor = Input('data.csv')
# input_processor.process_data()
# output_processor = Output(input_processor.df)
# output_processor.to_sqlite('database.db', 'table_name')
# output_processor.to_csv('output.csv')
# output_processor.to_parquet('output.parquet')