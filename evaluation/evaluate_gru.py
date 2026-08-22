import os
import sys
import torch
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


# ============================================================
# IMPORT GRU MODEL
# ============================================================

from models.gru_predictor import GRUPredictor


# ============================================================
# CONFIGURATION
# ============================================================

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

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "computer_metrics.csv"
)

SEQUENCE_LENGTH = 10

TEST_RATIO = 0.20


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

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


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 75)
print("GRU MODEL PERFORMANCE EVALUATION")
print("=" * 75)


print("\nLoading dataset...")

dataset = pd.read_csv(
    DATASET_PATH
)

print("Dataset loaded successfully!")

print(
    f"Total dataset samples: {len(dataset)}"
)


# ============================================================
# CHECK DATASET COLUMNS
# ============================================================

print("\nChecking required dataset columns...")


missing_columns = [

    column

    for column in FEATURE_COLUMNS

    if column not in dataset.columns

]


if missing_columns:

    print(
        "\nERROR: Required columns are missing!"
    )

    print(
        "Missing columns:"
    )

    for column in missing_columns:

        print(
            f"- {column}"
        )

    print(
        "\nAvailable dataset columns:"
    )

    for column in dataset.columns:

        print(
            f"- {column}"
        )

    sys.exit()


print(
    "All required columns found successfully!"
)


# ============================================================
# EXTRACT FEATURES AND TARGET
# ============================================================

print("\nPreparing evaluation data...")


feature_data = dataset[
    FEATURE_COLUMNS
].values.astype(
    np.float32
)


target_data = dataset[
    TARGET_COLUMN
].values.astype(
    np.float32
)


print(
    f"Feature shape: {feature_data.shape}"
)

print(
    f"Target shape: {target_data.shape}"
)


# ============================================================
# LOAD NORMALIZATION VALUES
# ============================================================

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


print(
    "Normalization data loaded successfully!"
)


# ============================================================
# NORMALIZE FEATURES
# ============================================================

normalized_features = (

    feature_data
    -
    feature_min

) / (

    feature_range
    +
    1e-8

)


# ============================================================
# CREATE SEQUENCES
# ============================================================

print(
    f"\nCreating sequences "
    f"(sequence length = {SEQUENCE_LENGTH})..."
)


X = []

y = []


for i in range(
    len(normalized_features)
    -
    SEQUENCE_LENGTH
):

    sequence = normalized_features[
        i:
        i + SEQUENCE_LENGTH
    ]

    target = target_data[
        i + SEQUENCE_LENGTH
    ]


    X.append(
        sequence
    )

    y.append(
        target
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
    f"Total sequences created: {len(X)}"
)


# ============================================================
# CREATE TEST DATA
# ============================================================

print(
    f"\nUsing last "
    f"{TEST_RATIO * 100:.0f}% "
    f"of data for evaluation..."
)


test_start = int(

    len(X)
    *
    (
        1
        -
        TEST_RATIO
    )

)


X_test = X[
    test_start:
]

y_test = y[
    test_start:
]


print(
    f"Test samples: {len(X_test)}"
)


# ============================================================
# LOAD TRAINED GRU MODEL
# ============================================================

print("\nLoading trained GRU model...")


model = GRUPredictor(

    input_size=8,
    hidden_size=64,
    num_layers=2

)


model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )

)


model.eval()


print(
    "Trained GRU model loaded successfully!"
)


# ============================================================
# CONVERT TEST DATA TO PYTORCH
# ============================================================

X_test_tensor = torch.tensor(

    X_test,

    dtype=torch.float32

)


# ============================================================
# MAKE PREDICTIONS
# ============================================================

print("\nRunning GRU model evaluation...")


with torch.no_grad():

    predicted_normalized = model(
        X_test_tensor
    )


predicted_normalized = (

    predicted_normalized
    .squeeze()
    .numpy()

)


# ============================================================
# CONVERT PREDICTIONS TO CELSIUS
# ============================================================

predictions = (

    predicted_normalized
    *
    target_range

) + target_min


# ============================================================
# CALCULATE ERROR METRICS
# ============================================================

mae = mean_absolute_error(

    y_test,
    predictions

)


rmse = np.sqrt(

    mean_squared_error(
        y_test,
        predictions
    )

)


r2 = r2_score(

    y_test,
    predictions

)


# ============================================================
# CALCULATE MAPE
# ============================================================

mape = np.mean(

    np.abs(

        (
            y_test
            -
            predictions
        )

        /

        (
            y_test
            +
            1e-8
        )

    )

) * 100


# ============================================================
# ACCURACY WITHIN TEMPERATURE TOLERANCE
# ============================================================

absolute_error = np.abs(

    y_test
    -
    predictions

)


accuracy_2c = np.mean(

    absolute_error
    <= 2

) * 100


accuracy_3c = np.mean(

    absolute_error
    <= 3

) * 100


accuracy_5c = np.mean(

    absolute_error
    <= 5

) * 100


# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_DIR = os.path.join(

    PROJECT_ROOT,
    "results"

)


os.makedirs(

    RESULTS_DIR,
    exist_ok=True

)


evaluation_file = os.path.join(

    RESULTS_DIR,
    "gru_evaluation_results.csv"

)


evaluation_results = pd.DataFrame({

    "Actual Temperature": y_test,

    "Predicted Temperature": predictions,

    "Absolute Error": absolute_error

})


evaluation_results.to_csv(

    evaluation_file,

    index=False

)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 75)

print(
    "GRU MODEL PERFORMANCE RESULTS"
)

print("=" * 75)


print(
    f"\nMAE  (Mean Absolute Error) : "
    f"{mae:.4f} °C"
)


print(
    f"RMSE (Root Mean Square Error): "
    f"{rmse:.4f} °C"
)


print(
    f"R² Score                    : "
    f"{r2:.4f}"
)


print(
    f"MAPE                        : "
    f"{mape:.2f}%"
)


print("\n" + "-" * 75)

print(
    "TEMPERATURE PREDICTION ACCURACY"
)

print("-" * 75)


print(
    f"Within ±2°C  : "
    f"{accuracy_2c:.2f}%"
)


print(
    f"Within ±3°C  : "
    f"{accuracy_3c:.2f}%"
)


print(
    f"Within ±5°C  : "
    f"{accuracy_5c:.2f}%"
)


print("\n" + "-" * 75)

print(
    "MODEL INTERPRETATION"
)

print("-" * 75)


if r2 >= 0.90:

    print(
        "Model Quality: EXCELLENT"
    )

elif r2 >= 0.75:

    print(
        "Model Quality: GOOD"
    )

elif r2 >= 0.50:

    print(
        "Model Quality: ACCEPTABLE"
    )

else:

    print(
        "Model Quality: NEEDS IMPROVEMENT"
    )


print(
    f"\nEvaluation results saved to:"
)

print(
    evaluation_file
)


print("\n" + "=" * 75)

print(
    "GRU MODEL EVALUATION COMPLETED SUCCESSFULLY!"
)

print("=" * 75)