# ============================================================
# THERMAL-AWARE MEC ENVIRONMENT
# DQN REINFORCEMENT LEARNING ENVIRONMENT
# ============================================================

import sys
import os

# Allow Python to access project folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.reward_system import RewardSystem


class MECEnvironment:

    def __init__(self):

        # ====================================================
        # THERMAL INFORMATION
        # ====================================================

        self.predicted_temperature = 81.73
        self.thermal_threshold = 85.0

        self.thermal_headroom = (
            self.thermal_threshold - self.predicted_temperature
        )

        # ====================================================
        # TASK INFORMATION
        # ====================================================

        self.task_cpu_cycles = 500.0
        self.task_input_size = 5.0
        self.task_deadline = 120.0

        # ====================================================
        # MEC SERVER LOADS
        # ====================================================

        self.edge1_load = 30.0
        self.edge2_load = 50.0

        # ====================================================
        # DQN ENVIRONMENT SETTINGS
        # ====================================================

        self.state_size = 7
        self.action_size = 4

        # ====================================================
        # ACTION DEFINITIONS
        # ====================================================

        self.actions = {
            0: "LOCAL",
            1: "EDGE 1",
            2: "EDGE 2",
            3: "CLOUD"
        }

        # ====================================================
        # REWARD SYSTEM
        # ====================================================

        self.reward_system = RewardSystem(
            thermal_threshold=self.thermal_threshold,
            deadline=self.task_deadline
        )

        # ====================================================
        # CURRENT STATE
        # ====================================================

        self.state = self.get_state()


    # ========================================================
    # CREATE DQN STATE
    # ========================================================

    def get_state(self):

        return [

            self.predicted_temperature,
            self.thermal_headroom,
            self.task_cpu_cycles,
            self.task_input_size,
            self.task_deadline,
            self.edge1_load,
            self.edge2_load

        ]


    # ========================================================
    # RESET ENVIRONMENT
    # ========================================================

    def reset(self):

        self.state = self.get_state()

        return self.state


    # ========================================================
    # CALCULATE LATENCY AND ENERGY
    # ========================================================

    def calculate_metrics(self, action):

        # -----------------------------------------------
        # ACTION 0 → LOCAL EXECUTION
        # -----------------------------------------------

        if action == 0:

            latency = 110.0
            energy = 8.0


        # -----------------------------------------------
        # ACTION 1 → EDGE 1
        # -----------------------------------------------

        elif action == 1:

            latency = 95.0
            energy = 4.5


        # -----------------------------------------------
        # ACTION 2 → EDGE 2
        # -----------------------------------------------

        elif action == 2:

            latency = 105.0
            energy = 5.5


        # -----------------------------------------------
        # ACTION 3 → CLOUD
        # -----------------------------------------------

        elif action == 3:

            latency = 140.0
            energy = 3.0


        else:

            raise ValueError("Invalid action selected!")

        return latency, energy


    # ========================================================
    # EXECUTE ONE DQN ACTION
    # ========================================================

    def step(self, action):

        # Get selected execution location
        action_name = self.actions[action]


        # Calculate system metrics
        latency, energy = self.calculate_metrics(action)


        # Check deadline constraint
        deadline_met = latency <= self.task_deadline


        # Check thermal constraint
        thermal_safe = (
            self.predicted_temperature
            <= self.thermal_threshold
        )


        # Calculate DQN reward
        reward = self.reward_system.calculate_reward(

            latency=latency,
            energy=energy,
            predicted_temperature=self.predicted_temperature
        )


        # Environment is completed after one decision
        done = True


        # Get next state
        next_state = self.get_state()


        # Store additional information
        info = {

            "action": action_name,
            "latency": latency,
            "energy": energy,
            "predicted_temperature": (
                self.predicted_temperature
            ),
            "thermal_headroom": (
                self.thermal_headroom
            ),
            "deadline_met": deadline_met,
            "thermal_safe": thermal_safe,
            "reward": reward

        }


        return next_state, reward, done, info


# ============================================================
# TEST THE MEC ENVIRONMENT
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("THERMAL-AWARE MEC ENVIRONMENT TEST")
    print("=" * 65)


    # Create environment
    environment = MECEnvironment()


    # Print initial state
    state = environment.reset()

    print("\nInitial DQN State:")

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


    print("\n" + "-" * 65)

    # ========================================================
    # TEST ALL EXECUTION ACTIONS
    # ========================================================

    print("TESTING ALL MEC OFFLOADING ACTIONS")

    for action in range(4):

        next_state, reward, done, info = (
            environment.step(action)
        )

        print("\nAction:", action)

        print(
            "Execution Location:",
            info["action"]
        )

        print(
            f"Latency: {info['latency']:.2f} ms"
        )

        print(
            f"Energy: {info['energy']:.2f} J"
        )

        print(
            "Deadline Met:",
            info["deadline_met"]
        )

        print(
            "Thermal Safe:",
            info["thermal_safe"]
        )

        print(
            f"DQN Reward: {reward:.2f}"
        )


    print("\n" + "=" * 65)
    print("MEC ENVIRONMENT TEST COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    