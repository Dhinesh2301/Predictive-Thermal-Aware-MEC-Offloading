# ============================================================
# REWARD SYSTEM FOR
# PREDICTIVE THERMAL-AWARE DQN MEC OFFLOADING
# ============================================================


class RewardSystem:

    def __init__(
        self,
        thermal_threshold=85.0,
        deadline=120.0
    ):

        # Maximum safe operating temperature
        self.thermal_threshold = thermal_threshold

        # Maximum allowed task execution latency
        self.deadline = deadline


    # ========================================================
    # CALCULATE THERMAL-AWARE REWARD
    # ========================================================

    def calculate_reward(

        self,

        latency,

        energy,

        predicted_temperature,

        post_temperature

    ):


        # Initial reward
        reward = 0.0


        # ====================================================
        # 1. DEADLINE CONSTRAINT
        # ====================================================

        if latency <= self.deadline:

            # Reward successful deadline completion
            reward += 30

        else:

            # Strong penalty for deadline violation
            reward -= 40


        # ====================================================
        # 2. ENERGY CONSUMPTION
        # ====================================================

        # Lower energy consumption is preferred
        reward -= energy * 2


        # ====================================================
        # 3. THERMAL SAFETY AFTER EXECUTION
        # ====================================================

        if post_temperature < 75:

            # Excellent thermal condition
            reward += 25


        elif post_temperature <= self.thermal_threshold:

            # Safe thermal condition
            reward += 15


        else:

            # Still above the safe threshold
            reward -= 25


        # ====================================================
        # 4. THERMAL MITIGATION
        # ====================================================

        temperature_change = (

            post_temperature

            -

            predicted_temperature

        )


        if temperature_change < 0:

            # Temperature successfully reduced
            reduction = abs(
                temperature_change
            )


            # Reward the amount of thermal reduction
            reward += reduction * 5


        elif temperature_change > 0:

            # Temperature increased
            increase = (
                temperature_change
            )


            # Penalize thermal increase
            reward -= increase * 5


        # ====================================================
        # 5. CRITICAL THERMAL PENALTY
        # ====================================================

        if predicted_temperature > self.thermal_threshold:

            # The system predicted a dangerous condition
            reward -= 10


            # Extra penalty only when the temperature
            # remains above the threshold after execution
            if post_temperature > self.thermal_threshold:

                reward -= 10


        # ====================================================
        # 6. FINAL REWARD
        # ====================================================

        return round(
            reward,
            2
        )


# ============================================================
# TEST THE REWARD SYSTEM
# ============================================================

if __name__ == "__main__":


    print(
        "=" * 70
    )

    print(
        "PREDICTIVE THERMAL-AWARE DQN REWARD SYSTEM TEST"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # CREATE REWARD SYSTEM
    # --------------------------------------------------------

    reward_system = RewardSystem(

        thermal_threshold=85.0,

        deadline=120.0

    )


    # --------------------------------------------------------
    # EXAMPLE VALUES
    # --------------------------------------------------------

    latency = 95.0

    energy = 4.5

    predicted_temperature = 88.85

    post_temperature = 86.60


    # --------------------------------------------------------
    # CALCULATE TEMPERATURE CHANGE
    # --------------------------------------------------------

    temperature_change = (

        post_temperature

        -

        predicted_temperature

    )


    # --------------------------------------------------------
    # CALCULATE REWARD
    # --------------------------------------------------------

    reward = (

        reward_system.calculate_reward(

            latency=latency,

            energy=energy,

            predicted_temperature=(
                predicted_temperature
            ),

            post_temperature=(
                post_temperature
            )

        )

    )


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print()


    print(
        f"Latency: "
        f"{latency:.2f} ms"
    )


    print(
        f"Energy Consumption: "
        f"{energy:.2f} J"
    )


    print(
        f"Predicted Temperature: "
        f"{predicted_temperature:.2f} °C"
    )


    print(
        f"Post Temperature: "
        f"{post_temperature:.2f} °C"
    )


    print(
        f"Temperature Change: "
        f"{temperature_change:.2f} °C"
    )


    print()

    print(
        "-" * 70
    )


    print(
        f"Final DQN Reward: "
        f"{reward:.2f}"
    )


    print(
        "-" * 70
    )


    # --------------------------------------------------------
    # THERMAL ANALYSIS
    # --------------------------------------------------------

    if post_temperature < predicted_temperature:

        print(
            "Thermal Result: "
            "TEMPERATURE REDUCED"
        )


    elif post_temperature > predicted_temperature:

        print(
            "Thermal Result: "
            "TEMPERATURE INCREASED"
        )


    else:

        print(
            "Thermal Result: "
            "NO TEMPERATURE CHANGE"
        )


    # --------------------------------------------------------
    # THERMAL SAFETY STATUS
    # --------------------------------------------------------

    if post_temperature <= 85.0:

        print(
            "Post Thermal Status: "
            "SAFE"
        )

    else:

        print(
            "Post Thermal Status: "
            "CRITICAL - MITIGATION STILL REQUIRED"
        )


    print()

    print(
        "=" * 70
    )


    print(
        "REWARD SYSTEM TEST COMPLETED SUCCESSFULLY!"
    )


    print(
        "=" * 70
    )