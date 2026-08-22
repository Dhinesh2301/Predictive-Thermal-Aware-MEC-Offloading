import torch
from models.gru_predictor import GRUTemperaturePredictor


# Create GRU model
model = GRUTemperaturePredictor()


# Example temperature sequence
# Batch size = 1
# Sequence length = 5
# One feature = temperature

temperature_sequence = torch.tensor(
    [
        [
            [38.0],
            [38.6],
            [39.1],
            [39.8],
            [40.4]
        ]
    ],
    dtype=torch.float32
)


# Predict temperature
prediction = model(temperature_sequence)


print("GRU Model Created Successfully!")
print("Input Temperature Sequence:")

for temp in temperature_sequence[0]:
    print(temp.item(), "°C")

print("\nPredicted Temperature:")
print(prediction.item(), "°C")