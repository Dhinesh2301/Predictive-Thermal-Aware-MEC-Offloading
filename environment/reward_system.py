# ============================================================
# REWARD SYSTEM FOR THERMAL-AWARE DQN MEC OFFLOADING
# ============================================================


class RewardSystem:

    def __init__(
        self,
        thermal_threshold=85.0,
        deadline=120.0
    ):

        self.thermal_threshold = thermal_threshold
        self.deadline = deadline


    def calculate_reward(
        self,
        latency,
        energy,
        predicted_temperature
    ):

        reward = 0.0


        # =====================================================
        # 1. LATENCY REWARD / PENALTY
        # =====================================================

        if latency <= self.deadline:

            # Task completed before deadline
            reward += 20

        else:

            # Task missed deadline
            reward -= 30


        # =====================================================
        # 2. ENERGY PENALTY
        # =====================================================

        # Lower energy consumption gives better reward
        reward -= energy * 2


        # =====================================================
        # 3. THERMAL SAFETY REWARD / PENALTY
        # =====================================================

        if predicted_temperature < 75:

            reward += 20

        elif predicted_temperature < self.thermal_threshold:

            reward += 5

        else:

            # Thermal threshold exceeded
            reward -= 50


        # =====================================================
        # 4. FINAL REWARD
        # =====================================================

        return round(reward, 2)


# ============================================================
# TEST THE REWARD SYSTEM
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("THERMAL-AWARE DQN REWARD SYSTEM")
    print("=" * 65)

    reward_system = RewardSystem(
        thermal_threshold=85.0,
        deadline=120.0
    )


    # Example system values

    latency = 95.0
    energy = 4.5
    predicted_temperature = 81.73


    reward = reward_system.calculate_reward(

        latency=latency,
        energy=energy,
        predicted_temperature=predicted_temperature
    )


    print(f"Latency:               {latency:.2f} ms")
    print(f"Energy Consumption:    {energy:.2f} J")
    print(f"Predicted Temperature: {predicted_temperature:.2f} °C")

    print("-" * 65)

    print(f"Final DQN Reward:      {reward}")

    print("=" * 65)