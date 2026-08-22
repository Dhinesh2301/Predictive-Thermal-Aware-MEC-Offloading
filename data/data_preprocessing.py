import pandas as pd
import numpy as np


# Load the dataset
def load_and_preprocess_data(file_path):

    print("\nLoading dataset...")

    # Read CSV file
    df = pd.read_csv(file_path)

    print("Dataset loaded successfully!")

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    print("\nDataset Columns:")
    print(df.columns.tolist())

    print("\nOriginal Dataset Shape:")
    print(df.shape)

    # Show missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Remove rows containing missing values
    df = df.dropna()

    # Remove duplicate rows
    df = df.drop_duplicates()

    print("\nDataset Shape After Cleaning:")
    print(df.shape)

    # Convert all possible columns to numeric
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except:
            pass

    # Find temperature-related columns
    temperature_columns = [
        column for column in df.columns
        if "temp" in column.lower()
        or "temperature" in column.lower()
    ]

    print("\nTemperature Related Columns:")
    print(temperature_columns)

    # Display first five rows
    print("\nCleaned Dataset Preview:")
    print(df.head())

    return df


# Run preprocessing directly
if __name__ == "__main__":

    dataset_path = "data/Computer_metrics.csv"

    processed_data = load_and_preprocess_data(dataset_path)

    print("\nPreprocessing completed successfully!")

    # Save cleaned dataset
    processed_data.to_csv(
        "data/processed_computer_metrics.csv",
        index=False
    )

    print(
        "\nCleaned dataset saved as:"
        " data/processed_computer_metrics.csv"
    )