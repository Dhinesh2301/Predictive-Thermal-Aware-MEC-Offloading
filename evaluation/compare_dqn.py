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
# THERMAL PARAMETERS
# ============================================================

THERMAL_THRESHOLD = 85.0


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
# ESTIMATE THERMAL EFFECT OF EXECUTION ACTION
#
# This represents the expected thermal behavior of each
# execution location after the task is assigned.
#
# LOCAL:
#   Device executes task locally → temperature increases
#
# EDGE 1 / EDGE 2:
#   Heavy computation is offloaded → device temperature
#   decreases due to reduced local CPU activity
#
# CLOUD:
#   Maximum remote offloading → strongest local thermal relief
# ============================================================

def calculate_post_execution_temperature(
    predicted_temperature,
    action,
    task_cycles
):

    # --------------------------------------------------------
    # Normalize task size for thermal calculation
    # --------------------------------------------------------

    task_factor = min(
        task_cycles / 1000.0,
        1.0
    )


    # --------------------------------------------------------
    # THERMAL CHANGE BASED ON EXECUTION LOCATION
    # --------------------------------------------------------

    if action == 0:

        # LOCAL EXECUTION
        #
        # More local CPU activity increases temperature.
        #
        # Temperature increase:
        # Base heat generation + task workload effect

        thermal_change = (
            1.5
            +
            (3.0 * task_factor)
        )


    elif action == 1:

        # EDGE 1 OFFLOADING
        #
        # Most computation is moved away from IoT device.
        # The device gets thermal relief.

        thermal_change = (
            -(
                1.0
                +
                (2.5 * task_factor)
            )
        )


    elif action == 2:

        # EDGE 2 OFFLOADING

        thermal_change = (
            -(
                1.2
                +
                (2.8 * task_factor)
            )
        )


    elif action == 3:

        # CLOUD OFFLOADING
        #
        # Local CPU computation is minimized.

        thermal_change = (
            -(
                1.5
                +
                (3.0 * task_factor)
            )
        )


    else:

        thermal_change = 0.0


    # --------------------------------------------------------
    # CALCULATE POST-EXECUTION TEMPERATURE
    # --------------------------------------------------------

    post_temperature = (
        predicted_temperature
        +
        thermal_change
    )


    # Prevent unrealistic negative temperature

    post_temperature = max(
        0.0,
        post_temperature
    )


    # --------------------------------------------------------
    # TEMPERATURE REDUCTION
    # --------------------------------------------------------

    temperature_reduction = (
        predicted_temperature
        -
        post_temperature
    )


    # --------------------------------------------------------
    # THERMAL HEADROOM AFTER EXECUTION
    # --------------------------------------------------------

    post_headroom = (
        THERMAL_THRESHOLD
        -
        post_temperature
    )


    # --------------------------------------------------------
    # THERMAL STATUS
    # --------------------------------------------------------

    if post_temperature < 50:

        thermal_status = "COOL"

    elif post_temperature < 70:

        thermal_status = "NORMAL"

    elif post_temperature < THERMAL_THRESHOLD:

        thermal_status = "HIGH"

    else:

        thermal_status = "CRITICAL"


    return {

        "predicted_temperature":
            predicted_temperature,

        "post_temperature":
            post_temperature,

        "temperature_change":
            thermal_change,

        "temperature_reduction":
            temperature_reduction,

        "thermal_headroom":
            post_headroom,

        "thermal_status":
            thermal_status
    }


# ============================================================
# PREPARE ENVIRONMENT STATE USING GRU TEMPERATURE
# ============================================================

def reset_environment_with_temperature(
    env,
    predicted_temperature
):

    # Reset environment normally

    state = env.reset()


    # --------------------------------------------------------
    # APPLY GRU PREDICTED TEMPERATURE
    # --------------------------------------------------------

    if predicted_temperature is not None:

        # State index 0 = Predicted Temperature

        state[0] = (
            predicted_temperature
        )


        # State index 1 = Thermal Headroom

        state[1] = (
            THERMAL_THRESHOLD
            -
            predicted_temperature
        )


        # ----------------------------------------------------
        # UPDATE ENVIRONMENT INTERNAL VARIABLES
        # ----------------------------------------------------

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
                THERMAL_THRESHOLD
                -
                predicted_temperature
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

    print(
        "PREDICTIVE THERMAL-AWARE DQN VS BASELINE METHODS"
    )

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


    current_temperature = (
        float(state[0])
    )


    thermal_headroom = (
        float(state[1])
    )


    task_cycles = (
        float(state[2])
    )


    # --------------------------------------------------------
    # CURRENT SYSTEM STATE
    # --------------------------------------------------------

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
        f"{current_temperature:.2f} °C"
    )


    print(
        f"Thermal Headroom      : "
        f"{thermal_headroom:.2f} °C"
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

        # ----------------------------------------------------
        # Reset environment
        # ----------------------------------------------------

        state = reset_environment_with_temperature(
            env,
            predicted_temperature
        )


        # ----------------------------------------------------
        # Execute baseline action
        # ----------------------------------------------------

        next_state, reward, done, info = (
            env.step(action)
        )


        # ----------------------------------------------------
        # CALCULATE THERMAL EFFECT
        # ----------------------------------------------------

        thermal_result = (
            calculate_post_execution_temperature(
                current_temperature,
                action,
                task_cycles
            )
        )


        result = {

            "method":
                ACTION_NAMES[action],

            "action":
                action,

            "latency":
                info["latency"],

            "energy":
                info["energy"],

            "deadline_met":
                info["deadline_met"],

            "thermal_safe":
                info["thermal_safe"],

            "reward":
                reward,

            "post_temperature":
                thermal_result[
                    "post_temperature"
                ],

            "temperature_reduction":
                thermal_result[
                    "temperature_reduction"
                ],

            "post_headroom":
                thermal_result[
                    "thermal_headroom"
                ],

            "thermal_status":
                thermal_result[
                    "thermal_status"
                ]
        }


        results.append(
            result
        )


        # ----------------------------------------------------
        # DISPLAY BASELINE RESULT
        # ----------------------------------------------------

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


        print(
            f"Post Temperature : "
            f"{thermal_result['post_temperature']:.2f} °C"
        )


        print(
            f"Temperature Change : "
            f"{thermal_result['temperature_change']:.2f} °C"
        )


        print(
            f"Thermal Status : "
            f"{thermal_result['thermal_status']}"
        )


    # ========================================================
    # DQN DECISION
    # ========================================================

    print(
        "\n" + "=" * 75
    )

    print(
        "TRAINED DQN DECISION"
    )

    print(
        "=" * 75
    )


    # --------------------------------------------------------
    # RESET ENVIRONMENT WITH GRU TEMPERATURE
    # --------------------------------------------------------

    state = reset_environment_with_temperature(
        env,
        predicted_temperature
    )


    # --------------------------------------------------------
    # CONVERT STATE TO PYTORCH TENSOR
    # --------------------------------------------------------

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
        env.step(
            dqn_action
        )
    )


    # --------------------------------------------------------
    # CALCULATE DQN THERMAL IMPROVEMENT
    # --------------------------------------------------------

    dqn_thermal = (
        calculate_post_execution_temperature(
            current_temperature,
            dqn_action,
            task_cycles
        )
    )


    # --------------------------------------------------------
    # DISPLAY DQN DECISION
    # --------------------------------------------------------

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


    print(
        "\n" + "-" * 75
    )

    print(
        "THERMAL MITIGATION ANALYSIS"
    )

    print(
        "-" * 75
    )


    print(
        f"Temperature Before Execution : "
        f"{current_temperature:.2f} °C"
    )


    print(
        f"Temperature After Execution  : "
        f"{dqn_thermal['post_temperature']:.2f} °C"
    )


    print(
        f"Temperature Reduction        : "
        f"{dqn_thermal['temperature_reduction']:.2f} °C"
    )


    print(
        f"Post Thermal Headroom        : "
        f"{dqn_thermal['thermal_headroom']:.2f} °C"
    )


    print(
        f"Post Thermal Status          : "
        f"{dqn_thermal['thermal_status']}"
    )


    # --------------------------------------------------------
    # THERMAL IMPROVEMENT STATUS
    # --------------------------------------------------------

    if (
        dqn_thermal[
            "temperature_reduction"
        ] > 0
    ):

        print(
            "\nThermal Improvement Status   : "
            "TEMPERATURE REDUCED"
        )

    elif (
        dqn_thermal[
            "temperature_reduction"
        ] < 0
    ):

        print(
            "\nThermal Improvement Status   : "
            "LOCAL EXECUTION HEAT GENERATED"
        )

    else:

        print(
            "\nThermal Improvement Status   : "
            "NO SIGNIFICANT CHANGE"
        )


    # ========================================================
    # FIND BEST BASELINE
    # ========================================================

    best_baseline = max(
        results,
        key=lambda x: x["reward"]
    )


    # ========================================================
    # FIND BEST THERMAL BASELINE
    # ========================================================

    best_thermal_baseline = min(
        results,
        key=lambda x: x["post_temperature"]
    )


    # ========================================================
    # FINAL PERFORMANCE COMPARISON
    # ========================================================

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


    print(
        f"\nBest Thermal Method : "
        f"{best_thermal_baseline['method']}"
    )


    print(
        f"Lowest Temperature  : "
        f"{best_thermal_baseline['post_temperature']:.2f} °C"
    )


    # ========================================================
    # PERFORMANCE IMPROVEMENT
    # ========================================================

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


    # ========================================================
    # DISPLAY Q VALUES
    # ========================================================

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


    # ========================================================
    # FINAL PROJECT VALIDATION
    # ========================================================

    print(
        "\n" + "=" * 75
    )

    print(
        "PREDICTIVE THERMAL-AWARE OFFLOADING VALIDATION"
    )

    print(
        "=" * 75
    )


    print(
        "\n1. Predictive Capability:"
    )

    print(
        "   GRU predicts the future CPU temperature "
        "before the offloading decision."
    )


    print(
        "\n2. Thermal Awareness:"
    )

    print(
        "   The predicted temperature is included in "
        "the DQN environment state."
    )


    print(
        "\n3. Thermal-Aware Action:"
    )

    print(
        f"   DQN selected: "
        f"{ACTION_NAMES[dqn_action]}"
    )


    print(
        "\n4. Thermal Mitigation:"
    )

    print(
        f"   Temperature change after decision: "
        f"{dqn_thermal['temperature_change']:.2f} °C"
    )


    print(
        "\n5. Deadline Constraint:"
    )

    print(
        f"   Deadline satisfied: "
        f"{dqn_info['deadline_met']}"
    )


    print(
        "\n6. Energy Optimization:"
    )

    print(
        f"   Energy consumption: "
        f"{dqn_info['energy']:.2f} J"
    )


    print(
        "\n" + "=" * 75
    )

    print(
        "PREDICTIVE THERMAL-AWARE MEC OFFLOADING "
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