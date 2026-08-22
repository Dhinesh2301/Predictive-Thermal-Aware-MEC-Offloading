# ============================================================
# PREDICTIVE THERMAL-AWARE MEC ENVIRONMENT
# DQN REINFORCEMENT LEARNING ENVIRONMENT
# ============================================================

import sys
import os

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from environment.reward_system import RewardSystem


# ============================================================
# MEC ENVIRONMENT CLASS
# ============================================================

class MECEnvironment:


    # ========================================================
    # INITIALIZE ENVIRONMENT
    # ========================================================

    def __init__(self):


        # ====================================================
        # THERMAL INFORMATION
        # ====================================================

        # Predicted temperature received from GRU
        self.predicted_temperature = 81.73

        # Maximum safe CPU temperature
        self.thermal_threshold = 85.0


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
        # INITIAL THERMAL HEADROOM
        # ====================================================

        self.thermal_headroom = (

            self.thermal_threshold

            -

            self.predicted_temperature

        )


        # ====================================================
        # CURRENT STATE
        # ====================================================

        self.state = self.get_state()


    # ========================================================
    # CREATE DQN STATE VECTOR
    # ========================================================

    def get_state(self):


        self.thermal_headroom = (

            self.thermal_threshold

            -

            self.predicted_temperature

        )


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


        # Update thermal headroom
        self.thermal_headroom = (

            self.thermal_threshold

            -

            self.predicted_temperature

        )


        # Create fresh state
        self.state = self.get_state()


        return self.state


    # ========================================================
    # CALCULATE LATENCY AND ENERGY
    # ========================================================

    def calculate_metrics(self, action):


        # ----------------------------------------------------
        # ACTION 0 -> LOCAL EXECUTION
        # ----------------------------------------------------

        if action == 0:

            latency = 110.0

            energy = 8.0


        # ----------------------------------------------------
        # ACTION 1 -> EDGE 1
        # ----------------------------------------------------

        elif action == 1:

            latency = 95.0

            energy = 4.5


        # ----------------------------------------------------
        # ACTION 2 -> EDGE 2
        # ----------------------------------------------------

        elif action == 2:

            latency = 105.0

            energy = 5.5


        # ----------------------------------------------------
        # ACTION 3 -> CLOUD
        # ----------------------------------------------------

        elif action == 3:

            latency = 140.0

            energy = 3.0


        else:

            raise ValueError(
                "Invalid action selected!"
            )


        return latency, energy


    # ========================================================
    # CALCULATE POST-EXECUTION TEMPERATURE
    # ========================================================

    def calculate_post_temperature(self, action):


        # ----------------------------------------------------
        # LOCAL EXECUTION
        #
        # Local execution increases device CPU load
        # and therefore increases temperature.
        # ----------------------------------------------------

        if action == 0:

            temperature_change = 3.0


        # ----------------------------------------------------
        # EDGE 1 OFFLOADING
        #
        # Offloading reduces local computation
        # and therefore allows temperature reduction.
        # ----------------------------------------------------

        elif action == 1:

            temperature_change = -2.25


        # ----------------------------------------------------
        # EDGE 2 OFFLOADING
        #
        # Slightly higher thermal reduction.
        # ----------------------------------------------------

        elif action == 2:

            temperature_change = -2.60


        # ----------------------------------------------------
        # CLOUD OFFLOADING
        #
        # Highest reduction because the least amount
        # of computation remains on the local device.
        # ----------------------------------------------------

        elif action == 3:

            temperature_change = -3.00


        else:

            raise ValueError(
                "Invalid action selected!"
            )


        # Calculate final device temperature
        post_temperature = (

            self.predicted_temperature

            +

            temperature_change

        )


        return (

            post_temperature,

            temperature_change

        )


    # ========================================================
    # DETERMINE THERMAL STATUS
    # ========================================================

    def get_thermal_status(
        self,
        temperature
    ):


        # ----------------------------------------------------
        # SAFE
        # ----------------------------------------------------

        if temperature < 80:

            return "SAFE"


        # ----------------------------------------------------
        # WARNING
        # ----------------------------------------------------

        elif temperature <= self.thermal_threshold:

            return "WARNING"


        # ----------------------------------------------------
        # CRITICAL
        # ----------------------------------------------------

        else:

            return "CRITICAL"


    # ========================================================
    # EXECUTE ONE DQN ACTION
    # ========================================================

    def step(self, action):


        # ----------------------------------------------------
        # GET EXECUTION LOCATION
        # ----------------------------------------------------

        action_name = self.actions[action]


        # ----------------------------------------------------
        # TEMPERATURE BEFORE EXECUTION
        # ----------------------------------------------------

        temperature_before = (

            self.predicted_temperature

        )


        # ----------------------------------------------------
        # CALCULATE LATENCY AND ENERGY
        # ----------------------------------------------------

        latency, energy = (

            self.calculate_metrics(
                action
            )

        )


        # ----------------------------------------------------
        # CALCULATE POST-EXECUTION TEMPERATURE
        # ----------------------------------------------------

        post_temperature, temperature_change = (

            self.calculate_post_temperature(
                action
            )

        )


        # ----------------------------------------------------
        # CALCULATE POST-EXECUTION THERMAL HEADROOM
        # ----------------------------------------------------

        post_thermal_headroom = (

            self.thermal_threshold

            -

            post_temperature

        )


        # ----------------------------------------------------
        # CHECK DEADLINE CONSTRAINT
        # ----------------------------------------------------

        deadline_met = (

            latency

            <=

            self.task_deadline

        )


        # ----------------------------------------------------
        # CHECK POST-EXECUTION THERMAL SAFETY
        # ----------------------------------------------------

        thermal_safe = (

            post_temperature

            <=

            self.thermal_threshold

        )


        # ----------------------------------------------------
        # DETERMINE THERMAL STATUS
        # ----------------------------------------------------

        thermal_status = (

            self.get_thermal_status(
                post_temperature
            )

        )


        # ----------------------------------------------------
        # CALCULATE UPDATED DQN REWARD
        # ----------------------------------------------------

        reward = (

            self.reward_system.calculate_reward(

                latency=latency,

                energy=energy,

                predicted_temperature=(
                    temperature_before
                ),

                post_temperature=(
                    post_temperature
                )

            )

        )


        # ----------------------------------------------------
        # CREATE NEXT STATE
        # ----------------------------------------------------

        # The next state stores the post-decision temperature.
        self.predicted_temperature = (

            post_temperature

        )


        self.thermal_headroom = (

            post_thermal_headroom

        )


        next_state = self.get_state()


        # ----------------------------------------------------
        # EPISODE COMPLETES AFTER ONE DECISION
        # ----------------------------------------------------

        done = True


        # ----------------------------------------------------
        # STORE ADDITIONAL INFORMATION
        # ----------------------------------------------------

        info = {


            "action": action_name,


            "latency": latency,


            "energy": energy,


            "temperature_before": (
                temperature_before
            ),


            "predicted_temperature": (
                temperature_before
            ),


            "post_temperature": (
                post_temperature
            ),


            "temperature_change": (
                temperature_change
            ),


            "thermal_headroom": (
                post_thermal_headroom
            ),


            "deadline_met": (
                deadline_met
            ),


            "thermal_safe": (
                thermal_safe
            ),


            "thermal_status": (
                thermal_status
            ),


            "reward": (
                reward
            )

        }


        return (

            next_state,

            reward,

            done,

            info

        )


# ============================================================
# TEST THE MEC ENVIRONMENT
# ============================================================

if __name__ == "__main__":


    print(
        "=" * 70
    )


    print(
        "PREDICTIVE THERMAL-AWARE MEC ENVIRONMENT TEST"
    )


    print(
        "=" * 70
    )


    # ========================================================
    # CREATE ENVIRONMENT
    # ========================================================

    environment = MECEnvironment()


    # ========================================================
    # GET INITIAL STATE
    # ========================================================

    state = environment.reset()


    print()

    print(
        "INITIAL DQN STATE"
    )

    print()


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


    print()

    print(
        "-" * 70
    )


    print(
        "TESTING ALL MEC OFFLOADING ACTIONS"
    )


    # ========================================================
    # TEST EACH ACTION
    # ========================================================

    for action in range(4):


        # IMPORTANT:
        # Reset the temperature before testing every action
        # so every method starts with the same conditions.

        environment.predicted_temperature = 81.73

        environment.reset()


        next_state, reward, done, info = (

            environment.step(
                action
            )

        )


        print()

        print(
            "=" * 70
        )


        print(
            f"ACTION: {action}"
        )


        print(
            f"EXECUTION LOCATION: "
            f"{info['action']}"
        )


        print(
            "=" * 70
        )


        print(
            f"Latency : "
            f"{info['latency']:.2f} ms"
        )


        print(
            f"Energy  : "
            f"{info['energy']:.2f} J"
        )


        print(
            f"Temperature Before : "
            f"{info['temperature_before']:.2f} °C"
        )


        print(
            f"Temperature After  : "
            f"{info['post_temperature']:.2f} °C"
        )


        print(
            f"Temperature Change : "
            f"{info['temperature_change']:.2f} °C"
        )


        print(
            f"Thermal Headroom   : "
            f"{info['thermal_headroom']:.2f} °C"
        )


        print(
            f"Thermal Status     : "
            f"{info['thermal_status']}"
        )


        print(
            f"Deadline Met       : "
            f"{info['deadline_met']}"
        )


        print(
            f"Thermal Safe       : "
            f"{info['thermal_safe']}"
        )


        print(
            f"DQN Reward         : "
            f"{reward:.2f}"
        )


    print()


    print(
        "=" * 70
    )


    print(
        "MEC ENVIRONMENT TEST COMPLETED SUCCESSFULLY!"
    )


    print(
        "=" * 70
    )