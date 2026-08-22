import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


from models.gru_predictor import GRUPredictor


# ==========================================================
# PROJECT PATHS
# ==========================================================

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "computer_metrics.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "gru_temperature_model.pth"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "gru_scaler.npz"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)


# ==========================================================
# CONFIGURATION
# ==========================================================

SEQUENCE_LENGTH = 10

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


# ==========================================================
# CREATE RESULTS DIRECTORY
# ==========================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 75)
print("PROFESSIONAL GRU EVALUATION VISUALIZATION")
print("=" * 75)

print("\nLoading dataset...")


df = pd.read_csv(
    DATASET_PATH
)

print("Dataset loaded successfully!")


# ==========================================================
# EXTRACT FEATURES AND TARGET
# ==========================================================

features = df[
    FEATURE_COLUMNS
].values.astype(
    np.float32
)

targets = df[
    TARGET_COLUMN
].values.astype(
    np.float32
)


print(
    f"Total dataset samples: "
    f"{len(df)}"
)


# ==========================================================
# LOAD NORMALIZATION DATA
# ==========================================================

print("\nLoading normalization data...")


scaler_data = np.load(
    SCALER_PATH
)


feature_min = scaler_data[
    "feature_min"
]

feature_range = scaler_data[
    "feature_range"
]


target_min = scaler_data[
    "target_min"
]

target_range = scaler_data[
    "target_range"
]


# ==========================================================
# NORMALIZE FEATURES
# ==========================================================

normalized_features = (
    features - feature_min
) / (
    feature_range + 1e-8
)


# ==========================================================
# CREATE SEQUENCES
# ==========================================================

print(
    f"\nCreating sequences "
    f"(length = {SEQUENCE_LENGTH})..."
)


X = []
y = []


for i in range(
    len(normalized_features)
    -
    SEQUENCE_LENGTH
):

    X.append(
        normalized_features[
            i:
            i + SEQUENCE_LENGTH
        ]
    )

    y.append(
        targets[
            i + SEQUENCE_LENGTH
        ]
    )


X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.float32
)


print(
    f"Total sequences: "
    f"{len(X)}"
)


# ==========================================================
# USE LAST 20% AS TEST DATA
# ==========================================================

split_index = int(
    len(X) * 0.8
)


X_test = X[
    split_index:
]

y_test = y[
    split_index:
]


print(
    f"Test samples: "
    f"{len(X_test)}"
)


# ==========================================================
# LOAD TRAINED GRU MODEL
# ==========================================================

print(
    "\nLoading trained GRU model..."
)


model = GRUPredictor(
    input_size=8,
    hidden_size=64,
    num_layers=2
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=torch.device(
            "cpu"
        )
    )
)


model.eval()


print(
    "GRU model loaded successfully!"
)


# ==========================================================
# RUN PREDICTION
# ==========================================================

print(
    "\nGenerating predictions..."
)


X_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)


with torch.no_grad():

    predictions_normalized = model(
        X_tensor
    ).numpy().flatten()


# ==========================================================
# DENORMALIZE PREDICTIONS
# ==========================================================

predictions = (
    predictions_normalized
    *
    target_range
) + target_min


# ==========================================================
# CALCULATE ERROR
# ==========================================================

errors = (
    predictions
    -
    y_test
)


absolute_errors = np.abs(
    errors
)


print(
    "Predictions generated successfully!"
)


# ==========================================================
# GRAPH 1
# ACTUAL VS PREDICTED TEMPERATURE
# ==========================================================

print(
    "\nCreating Actual vs Predicted graph..."
)


sample_size = min(
    300,
    len(y_test)
)


plt.figure(
    figsize=(14, 6)
)


plt.plot(
    y_test[:sample_size],
    label="Actual CPU Temperature",
    linewidth=2
)


plt.plot(
    predictions[:sample_size],
    label="GRU Predicted Temperature",
    linewidth=2
)


plt.title(
    "GRU: Actual vs Predicted CPU Temperature",
    fontsize=16,
    fontweight="bold"
)


plt.xlabel(
    "Test Sample"
)


plt.ylabel(
    "CPU Temperature (°C)"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


actual_predicted_path = os.path.join(
    RESULTS_DIR,
    "gru_actual_vs_predicted.png"
)


plt.savefig(
    actual_predicted_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"Saved: {actual_predicted_path}"
)


# ==========================================================
# GRAPH 2
# PREDICTION ERROR DISTRIBUTION
# ==========================================================

print(
    "\nCreating prediction error distribution..."
)


plt.figure(
    figsize=(10, 6)
)


plt.hist(
    errors,
    bins=40,
    edgecolor="black"
)


plt.axvline(
    0,
    linestyle="--",
    linewidth=2
)


plt.title(
    "GRU Temperature Prediction Error Distribution",
    fontsize=16,
    fontweight="bold"
)


plt.xlabel(
    "Prediction Error (°C)"
)


plt.ylabel(
    "Number of Predictions"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


error_distribution_path = os.path.join(
    RESULTS_DIR,
    "gru_error_distribution.png"
)


plt.savefig(
    error_distribution_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"Saved: {error_distribution_path}"
)


# ==========================================================
# GRAPH 3
# ABSOLUTE PREDICTION ERROR
# ==========================================================

print(
    "\nCreating absolute error graph..."
)


plt.figure(
    figsize=(14, 6)
)


plt.plot(
    absolute_errors[
        :sample_size
    ],
    linewidth=1.5
)


plt.axhline(
    np.mean(
        absolute_errors
    ),
    linestyle="--",
    linewidth=2,
    label=(
        f"Average Error = "
        f"{np.mean(absolute_errors):.2f} °C"
    )
)


plt.title(
    "GRU Absolute Temperature Prediction Error",
    fontsize=16,
    fontweight="bold"
)


plt.xlabel(
    "Test Sample"
)


plt.ylabel(
    "Absolute Error (°C)"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


absolute_error_path = os.path.join(
    RESULTS_DIR,
    "gru_absolute_error.png"
)


plt.savefig(
    absolute_error_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"Saved: {absolute_error_path}"
)


# ==========================================================
# GRAPH 4
# ACCURACY WITHIN TEMPERATURE RANGE
# ==========================================================

print(
    "\nCreating prediction tolerance graph..."
)


within_2 = np.mean(
    absolute_errors <= 2
) * 100


within_3 = np.mean(
    absolute_errors <= 3
) * 100


within_5 = np.mean(
    absolute_errors <= 5
) * 100


tolerance_labels = [
    "Within ±2°C",
    "Within ±3°C",
    "Within ±5°C"
]


accuracy_values = [
    within_2,
    within_3,
    within_5
]


plt.figure(
    figsize=(10, 6)
)


bars = plt.bar(
    tolerance_labels,
    accuracy_values
)


for bar, value in zip(
    bars,
    accuracy_values
):

    plt.text(
        bar.get_x()
        +
        bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )


plt.ylim(
    0,
    100
)


plt.title(
    "GRU Temperature Prediction Accuracy",
    fontsize=16,
    fontweight="bold"
)


plt.xlabel(
    "Prediction Tolerance"
)


plt.ylabel(
    "Predictions Within Range (%)"
)


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


accuracy_path = os.path.join(
    RESULTS_DIR,
    "gru_prediction_accuracy.png"
)


plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"Saved: {accuracy_path}"
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 75)
print(
    "GRU EVALUATION GRAPHS CREATED SUCCESSFULLY"
)
print("=" * 75)


print(
    "\nGraphs saved in results folder:"
)


print(
    "\n1. Actual vs Predicted:"
)

print(
    actual_predicted_path
)


print(
    "\n2. Error Distribution:"
)

print(
    error_distribution_path
)


print(
    "\n3. Absolute Prediction Error:"
)

print(
    absolute_error_path
)


print(
    "\n4. Prediction Accuracy:"
)

print(
    accuracy_path
)


print("\n" + "=" * 75)