import pandas as pd
import numpy as np
import os
import yaml 

from sklearn.model_selection import train_test_split

def load_params():
    return yaml.safe_load(open('params.yaml', 'r'))

def read_data(url):
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    df.drop(columns=['tweet_id'],inplace=True)
    final_df = df[df['sentiment'].isin(['happiness','sadness'])]
    final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)
    return final_df

def save_data(data_path, train_df, test_df):
    os.makedirs(data_path, exist_ok=True)
    train_df.to_csv(os.path.join(data_path, "train.csv"))
    test_df.to_csv(os.path.join(data_path, "test.csv"))

def main():
    params = load_params()
    test_size = params['data_ingestion']['test_size']
    df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    final_df = preprocess_data(df)

    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

    data_path = os.path.join("data","raw")
    save_data(data_path, train_data, test_data)

if __name__ == "__main__":
    main()
    # This will execute the main function when the script is run directly.
    # If this script is imported as a module, the main function will not execute automatically.
    # This is a common Python idiom to allow or prevent parts of code from being run when the modules are imported.