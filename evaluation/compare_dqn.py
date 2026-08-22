import os
import sys
import torch


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


from environment.mec_environment import MECEnvironment
from agents.dqn_agents import DQNAgent


# ============================================================
# PROJECT PATHS
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "dqn_offloading_model.pth"
)


GRU_TEMPERATURE_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "latest_temperature.txt"
)


# ============================================================
# ACTION NAMES
# ============================================================

ACTION_NAMES = {
    0: "LOCAL",
    1: "EDGE 1",
    2: "EDGE 2",
    3: "CLOUD"
}


# ============================================================
# LOAD LATEST GRU TEMPERATURE
# ============================================================

def load_latest_gru_temperature():

    try:

        if os.path.exists(
            GRU_TEMPERATURE_PATH
        ):

            with open(
                GRU_TEMPERATURE_PATH,
                "r"
            ) as file:

                temperature = float(
                    file.read().strip()
                )

                return temperature

    except Exception as e:

        print(
            f"Warning: Unable to load GRU temperature: {e}"
        )

    return None


# ============================================================
# PREPARE ENVIRONMENT STATE USING GRU TEMPERATURE
# ============================================================

def reset_environment_with_temperature(
    env,
    predicted_temperature
):

    # Reset environment normally
    state = env.reset()


    # If GRU temperature is available,
    # replace the temperature in the state
    if predicted_temperature is not None:

        # State index 0 = Predicted Temperature
        state[0] = predicted_temperature


        # State index 1 = Thermal Headroom
        state[1] = (
            85.0 - predicted_temperature
        )


        # ----------------------------------------------------
        # UPDATE ENVIRONMENT INTERNAL VARIABLES
        # ----------------------------------------------------

        # Different environment versions may use
        # different variable names.
        # We update them if they exist.

        if hasattr(
            env,
            "predicted_temperature"
        ):

            env.predicted_temperature = (
                predicted_temperature
            )


        if hasattr(
            env,
            "temperature"
        ):

            env.temperature = (
                predicted_temperature
            )


        if hasattr(
            env,
            "cpu_temperature"
        ):

            env.cpu_temperature = (
                predicted_temperature
            )


        if hasattr(
            env,
            "thermal_headroom"
        ):

            env.thermal_headroom = (
                85.0 - predicted_temperature
            )


        if hasattr(
            env,
            "current_state"
        ):

            env.current_state = state


    return state


# ============================================================
# MAIN FUNCTION
# ============================================================

def compare_dqn_with_baselines():

    print("=" * 75)
    print("THERMAL-AWARE DQN VS BASELINE METHODS")
    print("=" * 75)


    # --------------------------------------------------------
    # LOAD LATEST GRU TEMPERATURE
    # --------------------------------------------------------

    predicted_temperature = (
        load_latest_gru_temperature()
    )


    if predicted_temperature is not None:

        print(
            f"\nGRU Predicted Temperature Received: "
            f"{predicted_temperature:.2f} °C"
        )

    else:

        print(
            "\nWarning: No GRU temperature file found."
        )

        print(
            "Environment-generated temperature will be used."
        )


    # --------------------------------------------------------
    # CREATE MEC ENVIRONMENT
    # --------------------------------------------------------

    # IMPORTANT:
    # Do NOT pass predicted_temperature here.

    env = MECEnvironment()

    print(
        "\nMEC Environment Created Successfully!"
    )


    # --------------------------------------------------------
    # CREATE DQN AGENT
    # --------------------------------------------------------

    agent = DQNAgent(
        state_size=7,
        action_size=4
    )

    print(
        "DQN Agent Created Successfully!"
    )


    # --------------------------------------------------------
    # LOAD TRAINED DQN MODEL
    # --------------------------------------------------------

    print(
        "\nLoading trained DQN model..."
    )


    agent.model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu")
        )
    )


    agent.model.eval()


    print(
        "Trained DQN model loaded successfully!"
    )


    # --------------------------------------------------------
    # GET CURRENT SYSTEM STATE
    # --------------------------------------------------------

    state = reset_environment_with_temperature(
        env,
        predicted_temperature
    )


    print(
        "\n" + "-" * 75
    )

    print(
        "CURRENT SYSTEM STATE"
    )

    print(
        "-" * 75
    )


    print(
        f"Predicted Temperature : "
        f"{state[0]:.2f} °C"
    )


    print(
        f"Thermal Headroom      : "
        f"{state[1]:.2f} °C"
    )


    print(
        f"Task CPU Cycles       : "
        f"{state[2]:.2f} M"
    )


    print(
        f"Task Input Size       : "
        f"{state[3]:.2f} MB"
    )


    print(
        f"Task Deadline         : "
        f"{state[4]:.2f} ms"
    )


    print(
        f"Edge 1 Load           : "
        f"{state[5]:.2f} %"
    )


    print(
        f"Edge 2 Load           : "
        f"{state[6]:.2f} %"
    )


    # --------------------------------------------------------
    # BASELINE METHODS
    # --------------------------------------------------------

    results = []


    print(
        "\n" + "=" * 75
    )

    print(
        "BASELINE RESULTS"
    )

    print(
        "=" * 75
    )


    for action in range(4):

        # Reset environment and apply
        # the same GRU temperature
        state = reset_environment_with_temperature(
            env,
            predicted_temperature
        )


        # Execute baseline action
        next_state, reward, done, info = (
            env.step(action)
        )


        result = {

            "method": ACTION_NAMES[action],

            "action": action,

            "latency": info["latency"],

            "energy": info["energy"],

            "deadline_met": info["deadline_met"],

            "thermal_safe": info["thermal_safe"],

            "reward": reward
        }


        results.append(result)


        print(
            f"\nMethod: "
            f"{ACTION_NAMES[action]}"
        )


        print(
            f"Latency      : "
            f"{info['latency']:.2f} ms"
        )


        print(
            f"Energy       : "
            f"{info['energy']:.2f} J"
        )


        print(
            f"Deadline Met : "
            f"{info['deadline_met']}"
        )


        print(
            f"Thermal Safe : "
            f"{info['thermal_safe']}"
        )


        print(
            f"Reward       : "
            f"{reward:.2f}"
        )


    # --------------------------------------------------------
    # DQN DECISION
    # --------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print(
        "TRAINED DQN DECISION"
    )

    print(
        "=" * 75
    )


    # Reset environment with
    # the latest GRU prediction
    state = reset_environment_with_temperature(
        env,
        predicted_temperature
    )


    # Convert state to PyTorch tensor
    state_tensor = torch.FloatTensor(
        state
    ).unsqueeze(0)


    # --------------------------------------------------------
    # GET DQN Q VALUES
    # --------------------------------------------------------

    with torch.no_grad():

        q_values = agent.model(
            state_tensor
        )


        dqn_action = torch.argmax(
            q_values,
            dim=1
        ).item()


    # --------------------------------------------------------
    # EXECUTE DQN ACTION
    # --------------------------------------------------------

    state = reset_environment_with_temperature(
        env,
        predicted_temperature
    )


    next_state, dqn_reward, done, dqn_info = (
        env.step(dqn_action)
    )


    print(
        f"\nDQN Selected Action : "
        f"{dqn_action}"
    )


    print(
        f"Execution Location  : "
        f"{ACTION_NAMES[dqn_action]}"
    )


    print(
        f"Latency             : "
        f"{dqn_info['latency']:.2f} ms"
    )


    print(
        f"Energy              : "
        f"{dqn_info['energy']:.2f} J"
    )


    print(
        f"Deadline Met        : "
        f"{dqn_info['deadline_met']}"
    )


    print(
        f"Thermal Safe        : "
        f"{dqn_info['thermal_safe']}"
    )


    print(
        f"DQN Reward          : "
        f"{dqn_reward:.2f}"
    )


    # --------------------------------------------------------
    # FIND BEST BASELINE
    # --------------------------------------------------------

    best_baseline = max(
        results,
        key=lambda x: x["reward"]
    )


    # --------------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print(
        "FINAL PERFORMANCE COMPARISON"
    )

    print(
        "=" * 75
    )


    print(
        f"\nBest Baseline      : "
        f"{best_baseline['method']}"
    )


    print(
        f"Baseline Reward    : "
        f"{best_baseline['reward']:.2f}"
    )


    print(
        f"DQN Decision       : "
        f"{ACTION_NAMES[dqn_action]}"
    )


    print(
        f"DQN Reward         : "
        f"{dqn_reward:.2f}"
    )


    # --------------------------------------------------------
    # PERFORMANCE IMPROVEMENT
    # --------------------------------------------------------

    reward_difference = (
        dqn_reward
        -
        best_baseline["reward"]
    )


    print(
        "\n" + "-" * 75
    )

    print(
        "DQN PERFORMANCE ANALYSIS"
    )

    print(
        "-" * 75
    )


    print(
        f"Reward Difference  : "
        f"{reward_difference:.2f}"
    )


    if dqn_reward >= best_baseline["reward"]:

        print(
            "Result             : "
            "DQN selected an optimal or equal-best MEC decision."
        )

    else:

        print(
            "Result             : "
            "DQN decision is currently below the best baseline."
        )


    # --------------------------------------------------------
    # DISPLAY Q VALUES
    # --------------------------------------------------------

    print(
        "\n" + "-" * 75
    )

    print(
        "DQN Q-VALUES"
    )

    print(
        "-" * 75
    )


    q_values_array = (
        q_values
        .squeeze()
        .cpu()
        .numpy()
    )


    for i, value in enumerate(
        q_values_array
    ):

        print(
            f"{ACTION_NAMES[i]:<10} → "
            f"{value:.4f}"
        )


    print(
        "\n" + "=" * 75
    )

    print(
        "DQN VS BASELINE COMPARISON "
        "COMPLETED SUCCESSFULLY!"
    )

    print(
        "=" * 75
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    compare_dqn_with_baselines()