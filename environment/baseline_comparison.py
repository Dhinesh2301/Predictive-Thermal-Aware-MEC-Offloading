import os
import sys

# Allow importing from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.mec_environment import MECEnvironment


def compare_baselines():
    print("=" * 75)
    print("THERMAL-AWARE MEC OFFLOADING - BASELINE COMPARISON")
    print("=" * 75)

    # Create MEC environment
    env = MECEnvironment()

    # Get current system state
    state = env.reset()

    print("\nSystem State:")
    print(f"Predicted Temperature : {state[0]:.2f} °C")
    print(f"Thermal Headroom      : {state[1]:.2f} °C")
    print(f"Task CPU Cycles       : {state[2]:.2f} M")
    print(f"Task Input Size       : {state[3]:.2f} MB")
    print(f"Task Deadline         : {state[4]:.2f} ms")

    print("\n" + "-" * 75)
    print("COMPARING BASELINE METHODS")
    print("-" * 75)

    actions = {
        0: "LOCAL",
        1: "EDGE 1",
        2: "EDGE 2",
        3: "CLOUD"
    }

    results = []

    for action in actions:
        # Execute action in MEC environment
        next_state, reward, done, info = env.step(action)

        result = {
            "action": action,
            "location": actions[action],
            "latency": info["latency"],
            "energy": info["energy"],
            "deadline_met": info["deadline_met"],
            "thermal_safe": info["thermal_safe"],
            "reward": reward
        }

        results.append(result)

        print(f"\n{actions[action]}")
        print(f"Latency        : {info['latency']:.2f} ms")
        print(f"Energy         : {info['energy']:.2f} J")
        print(f"Deadline Met   : {info['deadline_met']}")
        print(f"Thermal Safe   : {info['thermal_safe']}")
        print(f"Reward         : {reward:.2f}")

        # Reset environment before testing next baseline
        env.reset()

    # Find best baseline based on reward
    best_baseline = max(results, key=lambda x: x["reward"])

    print("\n" + "=" * 75)
    print("BEST BASELINE RESULT")
    print("=" * 75)

    print(f"Best Location  : {best_baseline['location']}")
    print(f"Best Reward    : {best_baseline['reward']:.2f}")
    print(f"Latency        : {best_baseline['latency']:.2f} ms")
    print(f"Energy         : {best_baseline['energy']:.2f} J")

    print("\n" + "=" * 75)
    print("BASELINE COMPARISON COMPLETED SUCCESSFULLY!")
    print("=" * 75)

    return results


if __name__ == "__main__":
    results = compare_baselines()