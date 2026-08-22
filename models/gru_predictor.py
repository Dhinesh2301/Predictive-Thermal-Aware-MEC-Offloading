import torch
import torch.nn as nn


class GRUPredictor(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2
    ):

        super(GRUPredictor, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )


    def forward(self, x):

        # Pass the input sequence through GRU
        gru_output, hidden = self.gru(x)

        # Get the output of the final time step
        last_output = gru_output[:, -1, :]

        # Predict the CPU temperature
        prediction = self.fc(last_output)

        return prediction