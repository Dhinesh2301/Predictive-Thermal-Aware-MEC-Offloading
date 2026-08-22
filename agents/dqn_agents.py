# ============================================================
# DEEP Q-NETWORK (DQN) AGENT
# FOR THERMAL-AWARE MEC OFFLOADING
# ============================================================

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# NEURAL NETWORK
# ============================================================

class DQNNetwork(nn.Module):

    def __init__(self, state_size, action_size):

        super(DQNNetwork, self).__init__()

        # Input → Hidden Layer 1
        self.fc1 = nn.Linear(state_size, 128)

        # Hidden Layer 1 → Hidden Layer 2
        self.fc2 = nn.Linear(128, 64)

        # Hidden Layer 2 → Q-Values for actions
        self.fc3 = nn.Linear(64, action_size)


    def forward(self, state):

        # ReLU activation
        x = torch.relu(self.fc1(state))

        x = torch.relu(self.fc2(x))

        # Output Q-value for every action
        return self.fc3(x)


# ============================================================
# DQN AGENT
# ============================================================

class DQNAgent:

    def __init__(self, state_size, action_size):

        # State contains 7 values
        self.state_size = state_size

        # Four possible actions
        self.action_size = action_size


        # ----------------------------------------------------
        # DQN HYPERPARAMETERS
        # ----------------------------------------------------

        self.gamma = 0.95

        self.learning_rate = 0.001

        self.epsilon = 1.0

        self.epsilon_min = 0.01

        self.epsilon_decay = 0.995

        self.batch_size = 32


        # ----------------------------------------------------
        # EXPERIENCE REPLAY MEMORY
        # ----------------------------------------------------

        self.memory = deque(maxlen=2000)


        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        self.device = torch.device(

            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


        # ----------------------------------------------------
        # DQN MODEL
        # ----------------------------------------------------

        self.model = DQNNetwork(
            state_size,
            action_size
        ).to(self.device)


        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        self.optimizer = optim.Adam(

            self.model.parameters(),

            lr=self.learning_rate
        )


        # ----------------------------------------------------
        # LOSS FUNCTION
        # ----------------------------------------------------

        self.loss_function = nn.MSELoss()


    # ========================================================
    # SELECT ACTION
    # EPSILON-GREEDY STRATEGY
    # ========================================================

    def act(self, state):

        # Random action for exploration
        if np.random.rand() <= self.epsilon:

            return random.randrange(
                self.action_size
            )


        # Convert state into PyTorch tensor
        state_tensor = torch.FloatTensor(

            state

        ).unsqueeze(0).to(self.device)


        # Get action without gradient calculation
        with torch.no_grad():

            q_values = self.model(
                state_tensor
            )


        # Choose action with highest Q-value
        action = torch.argmax(

            q_values

        ).item()


        return action


    # ========================================================
    # STORE EXPERIENCE
    # ========================================================

    def remember(

        self,
        state,
        action,
        reward,
        next_state,
        done

    ):

        self.memory.append(

            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )


    # ========================================================
    # EXPERIENCE REPLAY
    # ========================================================

    def replay(self):

        # Do not train until enough experiences exist
        if len(self.memory) < self.batch_size:

            return None


        # Randomly select experiences
        batch = random.sample(

            self.memory,

            self.batch_size
        )


        # ----------------------------------------------------
        # CONVERT DATA INTO TENSORS
        # ----------------------------------------------------

        states = torch.FloatTensor(

            np.array([
                experience[0]
                for experience in batch
            ])

        ).to(self.device)


        actions = torch.LongTensor(

            [
                experience[1]
                for experience in batch
            ]

        ).to(self.device)


        rewards = torch.FloatTensor(

            [
                experience[2]
                for experience in batch
            ]

        ).to(self.device)


        next_states = torch.FloatTensor(

            np.array([
                experience[3]
                for experience in batch
            ])

        ).to(self.device)


        dones = torch.FloatTensor(

            [
                experience[4]
                for experience in batch
            ]

        ).to(self.device)


        # ----------------------------------------------------
        # CURRENT Q VALUES
        # ----------------------------------------------------

        current_q_values = self.model(

            states

        ).gather(

            1,

            actions.unsqueeze(1)

        ).squeeze(1)


        # ----------------------------------------------------
        # NEXT Q VALUES
        # ----------------------------------------------------

        with torch.no_grad():

            max_next_q_values = torch.max(

                self.model(next_states),

                dim=1

            )[0]


        # ----------------------------------------------------
        # CALCULATE TARGET Q VALUES
        # ----------------------------------------------------

        target_q_values = (

            rewards

            +

            self.gamma

            *

            max_next_q_values

            *

            (1 - dones)

        )


        # ----------------------------------------------------
        # CALCULATE LOSS
        # ----------------------------------------------------

        loss = self.loss_function(

            current_q_values,

            target_q_values
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


        # ----------------------------------------------------
        # DECAY EPSILON
        # ----------------------------------------------------

        if self.epsilon > self.epsilon_min:

            self.epsilon *= (

                self.epsilon_decay
            )


        return loss.item()


# ============================================================
# TEST DQN AGENT
# ============================================================

if __name__ == "__main__":

    print("=" * 65)

    print("DQN AGENT CREATED SUCCESSFULLY")

    print("=" * 65)


    agent = DQNAgent(

        state_size=7,

        action_size=4
    )


    print(

        f"State Size: "
        f"{agent.state_size}"
    )

    print(

        f"Action Size: "
        f"{agent.action_size}"
    )


    print("\nAvailable Actions:")

    print("0 → LOCAL")

    print("1 → EDGE 1")

    print("2 → EDGE 2")

    print("3 → CLOUD")


    # Test system state
    test_state = np.array(

        [
            81.73,
            3.27,
            500.0,
            5.0,
            120.0,
            30.0,
            50.0
        ],

        dtype=np.float32
    )


    print("\nTest State:")

    print(test_state)


    # DQN selects action
    action = agent.act(

        test_state
    )


    print(

        f"\nDQN Selected Action: "
        f"{action}"
    )

    print(

        f"Current Epsilon: "
        f"{agent.epsilon}"
    )