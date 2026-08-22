import os
import sys
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)


# ============================================================
# MEC OFFLOADING RESULTS
# ============================================================

methods = [
    "Local",
    "Edge 1",
    "Edge 2",
    "Cloud",
    "DQN Agent"
]

latency = [
    110,
    95,
    105,
    140,
    95
]

energy = [
    8.0,
    4.5,
    5.5,
    3.0,
    4.5
]

reward = [
    9,
    16,
    14,
    -31,
    16
]


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# LATENCY GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    methods,
    latency
)

plt.title(
    "MEC Offloading Latency Comparison"
)

plt.xlabel(
    "Execution Method"
)

plt.ylabel(
    "Latency (ms)"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

latency_path = os.path.join(
    RESULTS_DIR,
    "latency_comparison.png"
)

plt.savefig(
    latency_path,
    dpi=300
)

plt.close()


# ============================================================
# ENERGY GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    methods,
    energy
)

plt.title(
    "MEC Offloading Energy Consumption"
)

plt.xlabel(
    "Execution Method"
)

plt.ylabel(
    "Energy Consumption (J)"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

energy_path = os.path.join(
    RESULTS_DIR,
    "energy_comparison.png"
)

plt.savefig(
    energy_path,
    dpi=300
)

plt.close()


# ============================================================
# REWARD GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    methods,
    reward
)

plt.title(
    "DQN Reward Comparison"
)

plt.xlabel(
    "Execution Method"
)

plt.ylabel(
    "Reward"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

reward_path = os.path.join(
    RESULTS_DIR,
    "reward_comparison.png"
)

plt.savefig(
    reward_path,
    dpi=300
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("PROFESSIONAL EVALUATION GRAPHS CREATED SUCCESSFULLY")
print("=" * 70)

print("\nGraphs saved in results folder:")

print(
    f"\n1. Latency Comparison:\n"
    f"{latency_path}"
)

print(
    f"\n2. Energy Comparison:\n"
    f"{energy_path}"
)

print(
    f"\n3. Reward Comparison:\n"
    f"{reward_path}"
)

print("\n" + "=" * 70)