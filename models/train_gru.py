import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow Python to find gru_predictor.py
sys.path.append(os.path.dirname(__file__))

from gru_predictor import GRUPredictor


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

SEQUENCE_LENGTH = 10
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("        GRU TRAINING - CPU TEMPERATURE PREDICTION")
print("=" * 65)

print("\nDevice being used:", DEVICE)


# --------------------------------------------------
# 2. LOAD THE REAL PROCESSED DATASET
# --------------------------------------------------

DATA_PATH = "data/processed_computer_metrics.csv"

print("\nLoading processed dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. SELECT INPUT FEATURES
# --------------------------------------------------

FEATURE_COLUMNS = [
    "CPU_Usage",
    "CPU_Frequency",
    "Memory_Usage",
    "Disk_Usage",
    "Process_Count",
    "Thread_Count",
    "GPU_Temperature",
    "CPU_Temperature"
]

TARGET_COLUMN = "CPU_Temperature"

print("\nInput Features:")
for feature in FEATURE_COLUMNS:
    print("-", feature)

print("\nPrediction Target:")
print("-", TARGET_COLUMN)


# --------------------------------------------------
# 4. CONVERT DATA INTO NUMPY ARRAYS
# --------------------------------------------------

feature_data = df[FEATURE_COLUMNS].values.astype(np.float32)

target_data = df[TARGET_COLUMN].values.astype(np.float32)

print("\nFeature data shape:", feature_data.shape)
print("Target data shape:", target_data.shape)


# --------------------------------------------------
# 5. SPLIT DATA
# --------------------------------------------------

split_index = int(len(feature_data) * 0.80)

train_features = feature_data[:split_index]
test_features = feature_data[split_index:]

train_target = target_data[:split_index]
test_target = target_data[split_index:]

print("\nTraining samples:", len(train_features))
print("Testing samples:", len(test_features))


# --------------------------------------------------
# 6. NORMALIZE DATA
# --------------------------------------------------

# Feature normalization values calculated from
# training data only
feature_min = train_features.min(axis=0)
feature_max = train_features.max(axis=0)

feature_range = feature_max - feature_min

# Prevent division by zero
feature_range[feature_range == 0] = 1

train_features_scaled = (
    train_features - feature_min
) / feature_range

test_features_scaled = (
    test_features - feature_min
) / feature_range


# Target normalization
target_min = train_target.min()
target_max = train_target.max()

target_range = target_max - target_min

if target_range == 0:
    target_range = 1

train_target_scaled = (
    train_target - target_min
) / target_range

test_target_scaled = (
    test_target - target_min
) / target_range


print("\nData normalization completed!")


# --------------------------------------------------
# 7. CREATE SEQUENCES FOR GRU
# --------------------------------------------------

def create_sequences(features, targets, sequence_length):

    X = []
    y = []

    for i in range(len(features) - sequence_length):

        # Previous 10 system records
        sequence = features[
            i:i + sequence_length
        ]

        # Predict next CPU temperature
        next_temperature = targets[
            i + sequence_length
        ]

        X.append(sequence)
        y.append(next_temperature)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32)
    )


X_train, y_train = create_sequences(
    train_features_scaled,
    train_target_scaled,
    SEQUENCE_LENGTH
)

X_test, y_test = create_sequences(
    test_features_scaled,
    test_target_scaled,
    SEQUENCE_LENGTH
)


print("\nSequence creation completed!")

print("Training input shape:", X_train.shape)
print("Training target shape:", y_train.shape)

print("Testing input shape:", X_test.shape)
print("Testing target shape:", y_test.shape)


# --------------------------------------------------
# 8. CONVERT TO PYTORCH TENSORS
# --------------------------------------------------

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train).unsqueeze(1)

X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test).unsqueeze(1)


# --------------------------------------------------
# 9. CREATE DATA LOADER
# --------------------------------------------------

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# --------------------------------------------------
# 10. CREATE GRU MODEL
# --------------------------------------------------

INPUT_SIZE = len(FEATURE_COLUMNS)

model = GRUPredictor(
    input_size=INPUT_SIZE
).to(DEVICE)


print("\nGRU Model Created Successfully!")

print("\nModel Architecture:")
print(model)


# --------------------------------------------------
# 11. LOSS FUNCTION AND OPTIMIZER
# --------------------------------------------------

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# 12. TRAIN THE GRU
# --------------------------------------------------

print("\nStarting GRU training...")
print("-" * 65)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        # Forward pass
        predictions = model(batch_X)

        # Calculate error
        loss = criterion(
            predictions,
            batch_y
        )

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(train_loader)
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.6f}"
    )


# --------------------------------------------------
# 13. TEST THE MODEL
# --------------------------------------------------

model.eval()

with torch.no_grad():

    X_test_tensor = X_test_tensor.to(DEVICE)

    predictions = model(
        X_test_tensor
    ).cpu().numpy()


# Convert predictions back to Celsius
predictions_celsius = (
    predictions * target_range
) + target_min

actual_celsius = (
    y_test.reshape(-1, 1) * target_range
) + target_min


# --------------------------------------------------
# 14. CALCULATE ERROR
# --------------------------------------------------

mae = np.mean(
    np.abs(
        predictions_celsius - actual_celsius
    )
)

rmse = np.sqrt(
    np.mean(
        (
            predictions_celsius -
            actual_celsius
        ) ** 2
    )
)


print("\n" + "=" * 65)
print("              GRU TRAINING COMPLETED!")
print("=" * 65)

print(
    f"\nMean Absolute Error (MAE): "
    f"{mae:.4f} °C"
)

print(
    f"Root Mean Square Error (RMSE): "
    f"{rmse:.4f} °C"
)


# --------------------------------------------------
# 15. SAVE TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = "models/gru_temperature_model.pth"

torch.save(
    model.state_dict(),
    MODEL_PATH
)


# --------------------------------------------------
# 16. SAVE NORMALIZATION VALUES
# --------------------------------------------------

SCALER_PATH = "models/gru_scaler.npz"

np.savez(
    SCALER_PATH,

    feature_min=feature_min,
    feature_range=feature_range,

    target_min=target_min,
    target_range=target_range
)


print("\nTrained GRU model saved successfully!")
print("Model path:", MODEL_PATH)

print("\nNormalization data saved successfully!")
print("Scaler path:", SCALER_PATH)


# --------------------------------------------------
# 17. SHOW SAMPLE PREDICTIONS
# --------------------------------------------------

print("\n" + "=" * 65)
print("        SAMPLE CPU TEMPERATURE PREDICTIONS")
print("=" * 65)

for i in range(10):

    print(
        f"Sample {i + 1}: "
        f"Actual = {actual_celsius[i][0]:.2f} °C | "
        f"Predicted = {predictions_celsius[i][0]:.2f} °C"
    )