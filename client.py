import flwr as fl
import pandas as pd
import tensorflow as tf
import keras
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys

# Get the file name for this client's dataset from command-line arguments
client_data_file = sys.argv[1]  # e.g., Plant_Parameters_client_1.csv

# Load and preprocess the client-specific data
df = pd.read_csv(client_data_file)

# Encode and scale features
label_encoder = LabelEncoder()
df['Plant Type'] = label_encoder.fit_transform(df['Plant Type'])
cols = ['Phosphorus', 'Potassium', 'Urea',
        'T.S.P', 'M.O.P', 'Moisture', 'Temperature']
scaler = MinMaxScaler()
df[cols] = scaler.fit_transform(df[cols])

x = df.drop('Plant Type', axis=1)
y = df['Plant Type']

# Split the data for local training and testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)

# Define the model
model = keras.Sequential([
    # Update input shape based on number of features
    keras.layers.Dense(8, activation="relu", input_shape=(9,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(64, activation="relu"),


    # Update output layer based on number of classes
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=125, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


# Start the Flower cli~ent
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080", client=FlowerClient())
