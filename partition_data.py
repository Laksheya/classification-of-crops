import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("Plant_Parameters.csv")

# Define the number of clients
num_clients = 4


def split_stratified_into_four(df, class_column, num_clients=4):
    # Initialize empty DataFrames for each client
    client_dfs = [pd.DataFrame() for _ in range(num_clients)]

    # Get unique classes
    classes = df[class_column].unique()

    # For each class
    for cls in classes:
        # Get the data for this class
        class_data = df[df[class_column] == cls]

        # Calculate samples per client (approximately equal)
        samples_per_client = len(class_data) // num_clients

        # Shuffle the data
        class_data = class_data.sample(frac=1, random_state=42)

        # Split the data
        for i in range(num_clients):
            start_idx = i * samples_per_client
            if i == num_clients - 1:
                # Last client gets any remaining samples
                client_data = class_data.iloc[start_idx:]
            else:
                end_idx = (i + 1) * samples_per_client
                client_data = class_data.iloc[start_idx:end_idx]

            client_dfs[i] = pd.concat([client_dfs[i], client_data])

    return client_dfs


class_column = 'Plant Type'
client_datasets = split_stratified_into_four(df, class_column, num_clients)

# Save each client's dataset
for i, client_df in enumerate(client_datasets):
    # Shuffle the final dataset
    client_df = client_df.sample(frac=1, random_state=42)
    client_df.to_csv(f"Plant_Parameters_client_{i+1}.csv", index=False)
    print(f"Client {i+1} dataset shape:", client_df.shape)
    # Print class distribution for each client
    print(f"Client {i+1} class distribution:")
    print(client_df[class_column].value_counts())
    print()
