import os
import torch
import numpy as np

from models.gru_predictor import GRUPredictor


# =================================================
# CONFIGURATION
# =================================================

MODEL_PATH = "models/gru_temperature_model.pth"
SCALER_PATH = "models/gru_scaler.npz"

SEQUENCE_LENGTH = 10

# Path used to pass the latest GRU prediction
# to the DQN comparison script
RESULTS_DIRECTORY = "results"

TEMPERATURE_FILE = os.path.join(
    RESULTS_DIRECTORY,
    "latest_temperature.txt"
)


# =================================================
# LOAD TRAINED GRU MODEL
# =================================================

print("=" * 65)
print("       GRU CPU TEMPERATURE PREDICTION SYSTEM")
print("=" * 65)

print("\nLoading trained GRU model...")


# Create the same GRU architecture used during training
model = GRUPredictor(
    input_size=8,
    hidden_size=64,
    num_layers=2
)


# Load trained model weights
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )
)


# Set the model to evaluation mode
model.eval()


print("GRU model loaded successfully!")


# =================================================
# LOAD NORMALIZATION DATA
# =================================================

print("\nLoading normalization data...")


scaler_data = np.load(
    SCALER_PATH
)


# Load feature normalization values
feature_min = scaler_data[
    "feature_min"
]


feature_range = scaler_data[
    "feature_range"
]


# Load target normalization values
target_min = scaler_data[
    "target_min"
]


target_range = scaler_data[
    "target_range"
]


print(
    "Normalization data loaded successfully!"
)


# =================================================
# NORMALIZE INPUT
# =================================================

def normalize_input(data):

    normalized = (
        data - feature_min
    ) / (
        feature_range + 1e-8
    )

    return normalized


# =================================================
# PREDICT CPU TEMPERATURE
# =================================================

def predict_temperature(input_sequence):

    # Convert input sequence to NumPy array
    input_sequence = np.array(
        input_sequence,
        dtype=np.float32
    )


    # Normalize input
    normalized_input = normalize_input(
        input_sequence
    )


    # Convert to PyTorch tensor
    input_tensor = torch.tensor(
        normalized_input,
        dtype=torch.float32
    )


    # Add batch dimension
    # (10, 8) -> (1, 10, 8)
    input_tensor = input_tensor.unsqueeze(
        0
    )


    # Disable gradient calculation
    with torch.no_grad():

        prediction = model(
            input_tensor
        )


    # Extract normalized prediction
    predicted_normalized = (
        prediction.item()
    )


    # Convert prediction back to Celsius
    predicted_temperature = (
        predicted_normalized
        *
        target_range
    ) + target_min


    return float(
        predicted_temperature
    )


# =================================================
# GENERATE DYNAMIC SAMPLE INPUT
# =================================================

def generate_dynamic_sample_input():

    # Base system sequence
    base_input = np.array(
        [

            [40, 3500, 50, 70, 180, 2500, 45, 60],
            [45, 3500, 51, 70, 185, 2550, 46, 61],
            [50, 3600, 52, 71, 190, 2600, 48, 62],
            [55, 3600, 53, 71, 195, 2650, 49, 63],
            [60, 3700, 54, 72, 200, 2700, 50, 64],
            [65, 3700, 55, 72, 205, 2750, 52, 65],
            [70, 3800, 56, 73, 210, 2800, 53, 66],
            [75, 3800, 57, 73, 215, 2850, 54, 67],
            [80, 3900, 58, 74, 220, 2900, 55, 68],
            [85, 3900, 60, 74, 225, 2950, 57, 70]

        ],
        dtype=np.float32
    )


    # Generate different variation every execution
    cpu_usage_change = np.random.uniform(
        -8,
        8
    )


    cpu_frequency_change = np.random.uniform(
        -200,
        200
    )


    memory_change = np.random.uniform(
        -5,
        5
    )


    disk_change = np.random.uniform(
        -4,
        4
    )


    process_change = np.random.uniform(
        -25,
        25
    )


    thread_change = np.random.uniform(
        -250,
        250
    )


    gpu_temperature_change = np.random.uniform(
        -5,
        5
    )


    cpu_temperature_change = np.random.uniform(
        -8,
        8
    )


    # Apply variation to all 10 time steps
    variation = np.array(
        [

            cpu_usage_change,
            cpu_frequency_change,
            memory_change,
            disk_change,
            process_change,
            thread_change,
            gpu_temperature_change,
            cpu_temperature_change

        ],
        dtype=np.float32
    )


    dynamic_input = (
        base_input
        +
        variation
    )


    # =================================================
    # KEEP VALUES WITHIN REASONABLE LIMITS
    # =================================================

    dynamic_input[:, 0] = np.clip(
        dynamic_input[:, 0],
        1,
        100
    )


    dynamic_input[:, 1] = np.clip(
        dynamic_input[:, 1],
        1000,
        5000
    )


    dynamic_input[:, 2] = np.clip(
        dynamic_input[:, 2],
        1,
        100
    )


    dynamic_input[:, 3] = np.clip(
        dynamic_input[:, 3],
        1,
        100
    )


    dynamic_input[:, 4] = np.clip(
        dynamic_input[:, 4],
        1,
        1000
    )


    dynamic_input[:, 5] = np.clip(
        dynamic_input[:, 5],
        1,
        10000
    )


    dynamic_input[:, 6] = np.clip(
        dynamic_input[:, 6],
        20,
        100
    )


    dynamic_input[:, 7] = np.clip(
        dynamic_input[:, 7],
        20,
        100
    )


    return dynamic_input


# =================================================
# GENERATE INPUT
# =================================================

print(
    "\nGenerating dynamic CPU system data..."
)


sample_input = (
    generate_dynamic_sample_input()
)


print(
    "Dynamic system input generated successfully!"
)


# =================================================
# MAKE GRU PREDICTION
# =================================================

print(
    "\nGenerating CPU temperature prediction..."
)


predicted_temperature = (
    predict_temperature(
        sample_input
    )
)


# =================================================
# DETERMINE THERMAL STATUS
# =================================================

if predicted_temperature < 50:

    thermal_status = "COOL"


elif predicted_temperature < 70:

    thermal_status = "NORMAL"


elif predicted_temperature < 85:

    thermal_status = "WARM"


else:

    thermal_status = "HOT"


# =================================================
# SAVE LATEST TEMPERATURE FOR DQN
# =================================================

os.makedirs(
    RESULTS_DIRECTORY,
    exist_ok=True
)


with open(
    TEMPERATURE_FILE,
    "w"
) as file:

    file.write(
        f"{predicted_temperature:.6f}"
    )


print(
    f"\nLatest temperature saved to: "
    f"{TEMPERATURE_FILE}"
)


# =================================================
# DISPLAY RESULT
# =================================================

print(
    "\n" + "=" * 65
)


print(
    "             TEMPERATURE PREDICTION RESULT"
)


print(
    "=" * 65
)


print(
    f"\nPredicted CPU Temperature: "
    f"{predicted_temperature:.2f} °C"
)


print(
    f"Thermal Status: "
    f"{thermal_status}"
)


print(
    "\nPrediction completed successfully!"
)


print(
    "=" * 65
)