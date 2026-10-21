import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Example preprocessing: drop missing values
    return df.dropna()

def save_data(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    data = load_data('data/input_data.csv')  # Adjust the path as needed
    processed_data = preprocess_data(data)
    save_data(processed_data, 'data/processed_data.csv')