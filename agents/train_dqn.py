# ============================================================
# DQN TRAINING FOR THERMAL-AWARE MEC OFFLOADING
# ============================================================

import sys
import os
import numpy as np
import torch

# ------------------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# ------------------------------------------------------------

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from environment.mec_environment import MECEnvironment
from agents.dqn_agents import DQNAgent


# ============================================================
# SETTINGS
# ============================================================

EPISODES = 1000

MODEL_PATH = "models/dqn_offloading_model.pth"


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_dqn():

    print("=" * 70)
    print("THERMAL-AWARE DQN TRAINING FOR MEC OFFLOADING")
    print("=" * 70)


    # --------------------------------------------------------
    # CREATE ENVIRONMENT
    # --------------------------------------------------------

    environment = MECEnvironment()

    print("\nMEC Environment Created Successfully!")


    # --------------------------------------------------------
    # CREATE DQN AGENT
    # --------------------------------------------------------

    agent = DQNAgent(
        state_size=environment.state_size,
        action_size=environment.action_size
    )

    print("DQN Agent Created Successfully!")


    # --------------------------------------------------------
    # STORE TRAINING HISTORY
    # --------------------------------------------------------

    rewards_history = []


    print("\nStarting DQN Training...")
    print("-" * 70)


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for episode in range(1, EPISODES + 1):

        # Reset environment
        state = environment.reset()

        # Convert state to NumPy array
        state = np.array(
            state,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # DQN SELECTS ACTION
        # ----------------------------------------------------

        action = agent.act(state)


        # ----------------------------------------------------
        # ENVIRONMENT EXECUTES ACTION
        # ----------------------------------------------------

        next_state, reward, done, info = (
            environment.step(action)
        )


        # Convert next state
        next_state = np.array(
            next_state,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # STORE EXPERIENCE
        # ----------------------------------------------------

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )


        # ----------------------------------------------------
        # TRAIN DQN
        # ----------------------------------------------------

        loss = agent.replay()


        # Store reward
        rewards_history.append(reward)


        # ----------------------------------------------------
        # PRINT TRAINING PROGRESS
        # ----------------------------------------------------

        if episode == 1 or episode % 50 == 0:

            average_reward = np.mean(
                rewards_history[-50:]
            )

            print(
                f"Episode [{episode}/{EPISODES}] | "
                f"Action: {info['action']} | "
                f"Reward: {reward:.2f} | "
                f"Average Reward: {average_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )


    # ========================================================
    # SAVE TRAINED MODEL
    # ========================================================

    os.makedirs(
        "models",
        exist_ok=True
    )


    torch.save(
        agent.model.state_dict(),
        MODEL_PATH
    )


    print("\n" + "=" * 70)
    print("DQN TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    print(
        f"\nTrained DQN model saved successfully!"
    )

    print(
        f"Model path: {MODEL_PATH}"
    )


    # ========================================================
    # FINAL TEST
    # ========================================================

    print("\n" + "-" * 70)
    print("FINAL DQN DECISION TEST")
    print("-" * 70)


    state = environment.reset()

    state = np.array(
        state,
        dtype=np.float32
    )


    # Disable exploration for final decision
    original_epsilon = agent.epsilon

    agent.epsilon = 0.0

    action = agent.act(state)

    next_state, reward, done, info = (
        environment.step(action)
    )


    print(
        f"\nDQN Selected Action: {action}"
    )

    print(
        f"Execution Location: {info['action']}"
    )

    print(
        f"Latency: {info['latency']:.2f} ms"
    )

    print(
        f"Energy: {info['energy']:.2f} J"
    )

    print(
        f"DQN Reward: {reward:.2f}"
    )


    print("\n" + "=" * 70)


# ============================================================
# RUN TRAINING
# ============================================================

if __name__ == "__main__":

    train_dqn()