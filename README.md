# Predictive Thermal-Aware Task Offloading for Deadline-Constrained IoT Applications in Mobile Edge Computing

<p align="center">
  <strong>GRU-Based Temperature Prediction | Deep Q-Network | Mobile Edge Computing</strong>
</p>

<p align="center">
  A predictive and intelligent task offloading framework designed to balance thermal conditions, execution latency, energy consumption, and deadline requirements in IoT-based Mobile Edge Computing environments.
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Proposed Solution](#-proposed-solution)
- [System Architecture](#-system-architecture)
- [How the System Works](#-how-the-system-works)
- [Key Features](#-key-features)
- [Technologies Used](#-technologies-used)
- [Machine Learning and Deep Learning Techniques](#-machine-learning-and-deep-learning-techniques)
- [Task Offloading Strategies](#-task-offloading-strategies)
- [System Configuration](#-system-configuration)
- [Project Structure](#-project-structure)
- [Dataset Description](#-dataset-description)
- [Performance Evaluation](#-performance-evaluation)
- [Real-World Applications](#-real-world-applications)
- [Installation and Setup](#-installation-and-setup)
- [Execution Workflow](#-execution-workflow)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Contributors](#-contributors)
- [License](#-license)

---

## 🚀 Overview

Mobile Edge Computing (MEC) brings computational resources closer to IoT devices, reducing communication latency and enabling faster task execution. However, resource-constrained IoT devices may experience increasing CPU temperatures when handling computationally intensive workloads.

Excessive temperature can trigger **thermal throttling**, reducing processor frequency and increasing task execution time. This can negatively affect applications that require strict deadline compliance.

This project introduces a **Predictive Thermal-Aware Task Offloading Framework** that combines a Gated Recurrent Unit (GRU) network and a Deep Q-Network (DQN) to make intelligent task execution decisions.

The GRU model predicts future CPU temperature using historical system metrics. The predicted temperature is then used to calculate thermal headroom. Based on thermal conditions, task requirements, deadlines, network conditions, and edge-server availability, the DQN agent selects an appropriate execution destination.

The framework evaluates four execution options:

- Local Device
- Edge Server 1
- Edge Server 2
- Cloud Server

The proposed system is developed and evaluated using a controlled software-based MEC simulation environment.

## ❗ Problem Statement

Traditional task offloading approaches generally focus on minimizing execution latency, communication overhead, or energy consumption. However, they may not adequately consider the future thermal condition of an IoT device.

When a device continuously processes computationally intensive tasks:

1. CPU utilization increases.
2. Processor power consumption increases.
3. CPU temperature rises.
4. Thermal throttling may occur.
5. CPU frequency may decrease.
6. Task execution time may increase.
7. Deadline violations may occur.

Reactive thermal-aware systems generally respond to the current temperature. Such approaches may not anticipate a temperature increase before it occurs.

Therefore, an intelligent task offloading mechanism is required to predict future thermal conditions and make proactive decisions while considering task deadlines, latency, energy consumption, and available computing resources.

## 🎯 Project Objectives

- Predict future CPU temperature using historical system performance data.
- Calculate thermal headroom to identify potential thermal risks.
- Develop an intelligent task offloading mechanism using Deep Q-Learning.
- Select suitable execution destinations among local, edge, and cloud resources.
- Consider task deadlines during offloading decisions.
- Evaluate latency, energy consumption, thermal conditions, and task completion performance.
- Compare predictive thermal-aware offloading against conventional baseline strategies.

## 💡 Proposed Solution

The proposed framework integrates two major techniques:

### 1. GRU-Based Thermal Prediction

A Gated Recurrent Unit (GRU) model analyzes sequential system metrics and predicts future CPU temperature.

The prediction helps identify possible thermal risks before the device reaches its configured temperature threshold.

### 2. DQN-Based Intelligent Task Offloading

A Deep Q-Network (DQN) agent selects the appropriate task execution destination based on the system state.

The decision-making process considers:

- Current CPU temperature
- Predicted future temperature
- Thermal headroom
- CPU utilization
- Task computational requirements
- Task deadline
- Network conditions
- Edge-server workload

The overall objective is to make proactive offloading decisions while balancing execution performance, energy consumption, and thermal risk.

## 🏗️ System Architecture

The system follows a sequential architecture in which task generation, thermal prediction, decision-making, task execution, and performance evaluation are integrated.

```text
              ┌─────────────────────────┐
              │   IoT Task Generation   │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ System State Collection │
              │ CPU / Memory / Workload │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   GRU Temperature       │
              │      Prediction         │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Thermal Headroom        │
              │     Calculation         │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ DQN Offloading Agent    │
              │ Intelligent Decision   │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Task Execution        │
              └────────────┬────────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
       ┌──────────┐  ┌───────────┐  ┌───────────┐
       │  Local   │  │   Edge    │  │   Cloud   │
       │  Device  │  │  Servers  │  │  Server   │
       └──────────┘  └───────────┘  └───────────┘
             │             │             │
             └─────────────┼─────────────┘
                           ▼
              ┌─────────────────────────┐
              │ Performance Evaluation  │
              │ Latency / Energy / Temp │
              │ Deadline / Task Success │
              └─────────────────────────┘
```

## ⚙️ How the System Works

### Step 1: IoT Task Generation

IoT devices generate computational tasks with different requirements, including:

- Required CPU cycles
- Input data size
- Output data size
- Execution deadline

### Step 2: System State Collection

The system collects relevant system information, such as CPU utilization, CPU frequency, memory usage, and current CPU temperature.

These metrics help describe the current computational and thermal conditions.

### Step 3: Future Temperature Prediction

The GRU model processes historical system information and predicts the future CPU temperature.

This prediction provides an early indication of possible thermal stress.

### Step 4: Thermal Headroom Calculation

Thermal headroom represents the difference between the configured safe temperature threshold and the predicted future temperature.

**Formula:**

```text
Thermal Headroom = Safe Temperature - Predicted Temperature
```

For example:

```text
Configured Temperature Threshold = 85°C
Predicted Temperature             = 81°C

Thermal Headroom = 85 - 81
                 = 4°C
```

A smaller positive headroom indicates that the device is approaching its configured thermal threshold.

### Step 5: DQN-Based Offloading Decision

The DQN agent receives the system state and selects one of the available execution destinations.

The agent learns a decision-making policy through interactions with the MEC simulation environment and reward feedback.

### Step 6: Task Execution

The selected destination executes the task within the simulated environment.

The system estimates execution latency, energy consumption, and deadline compliance.

### Step 7: Performance Evaluation

The system records performance metrics and uses the results to evaluate the effectiveness of the offloading strategy.

## ✨ Key Features

| Feature | Description |
|---|---|
| Predictive Thermal Monitoring | Forecasts future CPU temperature using GRU. |
| Thermal Headroom Calculation | Estimates the remaining margin before the configured temperature threshold. |
| Intelligent Task Offloading | Uses DQN to select an execution destination. |
| Deadline-Aware Decisions | Considers task execution deadlines. |
| Multi-Server MEC Simulation | Supports local, multiple edge servers, and cloud execution. |
| Thermal-Aware Evaluation | Measures thermal conditions and potential throttling. |
| Baseline Comparison | Compares different offloading strategies. |
| Performance Visualization | Displays evaluation results through charts and dashboards. |

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| PyTorch | GRU and DQN model development |
| NumPy | Numerical computations |
| Pandas | Dataset processing and manipulation |
| Gymnasium | Simulation environment |
| Scikit-learn | Data preprocessing and evaluation utilities, where applicable |
| Matplotlib | Performance visualization |
| Streamlit | Interactive project dashboard |

## 🧠 Machine Learning and Deep Learning Techniques

### GRU – Gated Recurrent Unit

GRU is a recurrent neural network architecture designed to learn sequential patterns in time-series data.

In this project, GRU is used to:

- Learn temperature patterns from historical system metrics.
- Capture temporal relationships between workload and CPU temperature.
- Predict future CPU temperature.
- Provide predictive thermal information to the offloading agent.

**Model Evaluation Metrics:**

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### DQN – Deep Q-Network

Deep Q-Network is a reinforcement learning algorithm that combines Q-learning with a neural network to estimate action values.

In this project, DQN is responsible for selecting the task execution destination.

**State Information:**

- Current CPU temperature
- Predicted CPU temperature
- Thermal headroom
- CPU utilization
- Task CPU requirements
- Task deadline
- Network conditions
- Edge-server load

**Available Actions:**

```text
0 - Local Execution
1 - Edge Server 1
2 - Edge Server 2
3 - Cloud Execution
```

The agent learns to select actions based on the reward received from the simulation environment.

## 🔄 Task Offloading Strategies

The proposed framework supports comparison against multiple baseline approaches.

| Strategy | Description |
|---|---|
| Local Execution | Executes every task on the IoT device. |
| Latency-Oriented Offloading | Selects an execution destination based on estimated latency. |
| Reactive Thermal-Aware Offloading | Considers the current thermal condition. |
| DQN Without Predictive Thermal Information | Uses reinforcement learning without GRU-based future temperature information. |
| Proposed GRU + DQN | Uses predicted temperature and thermal headroom during intelligent offloading decisions. |

The comparison helps investigate whether predictive thermal information improves task offloading performance under different workload conditions.

## 🖥️ System Configuration

The simulation environment consists of the following components:

| Component | Configuration |
|---|---|
| IoT Devices | 3 |
| Edge Servers | 2 |
| Cloud Server | 1 |
| Execution Options | Local, Edge 1, Edge 2, Cloud |
| Thermal Prediction Model | GRU |
| Offloading Algorithm | DQN |
| Thermal Simulation | First-Order Thermal Model |
| Temperature Threshold | 85°C (configured simulation parameter) |

**Note:** The temperature threshold is a configurable simulation parameter and is not a universal safe operating limit for all processors.

## 📂 Project Structure

```text
Predictive-Thermal-Aware-MEC/
│
├── data/
│   ├── Computer_metrics.csv
│   ├── Computer_metrics_100k_balanced_thermal.csv
│   └── data_preprocessing.py
│
├── models/
│   ├── train_gru.py
│   ├── gru_predictor.py
│   └── gru_temperature_model.pth
│
├── agents/
│   ├── dqn_agents.py
│   └── train_dqn.py
│
├── environment/
│   ├── mec_environment.py
│   ├── thermal_manager.py
│   ├── reward_system.py
│   └── baseline_comparison.py
│
├── evaluation/
│   ├── evaluate_gru.py
│   ├── visualize_gru_evaluation.py
│   ├── compare_dqn.py
│   └── visualize_results.py
│
├── app.py
├── requirements.txt
└── README.md
```

*The structure above represents the intended project organization. Generated model files and processed datasets should be included in the repository only if they are available and suitable for sharing.*

## 📊 Dataset Description

The project uses computer performance metrics for thermal prediction.

The dataset contains system-level information such as:

| Attribute | Description |
|---|---|
| Timestamp | Time at which the system metric was recorded |
| CPU Usage | CPU utilization percentage |
| CPU Frequency | Operating CPU frequency |
| Memory Usage | Memory utilization |
| Disk Usage | Disk utilization |
| Process Count | Number of active processes |
| Thread Count | Number of active threads |
| CPU Temperature | Recorded CPU temperature |
| GPU Temperature | Recorded GPU temperature |

### Data Preprocessing

The preprocessing stage includes:

- Loading the system metrics dataset.
- Handling missing or invalid values.
- Preparing sequential time-series samples.
- Normalizing relevant input features.
- Preparing training and testing datasets.
- Generating inputs for GRU-based temperature prediction.

**Dataset Note:** If an augmented or synthetic balanced dataset is used, its generated temperature values must be identified as simulated data rather than independent real-world hardware measurements.

## 📈 Performance Evaluation

The proposed framework is evaluated using the following metrics.

### 1. Temperature Prediction Accuracy

| Metric | Purpose |
|---|---|
| MAE | Measures average absolute prediction error. |
| RMSE | Measures the square root of the average squared prediction error. |

### 2. Task Execution Performance

| Metric | Purpose |
|---|---|
| Average Latency | Measures average task completion time. |
| P95 Latency | Measures the 95th percentile task completion time. |
| Energy Consumption | Estimates energy used during task execution. |
| Task Completion Rate | Measures the percentage of successfully completed tasks. |
| Deadline Violations | Counts tasks that exceed their deadlines. |

### 3. Thermal Performance

| Metric | Purpose |
|---|---|
| Peak Temperature | Measures the maximum simulated CPU temperature. |
| Thermal Throttling Events | Counts simulated throttling occurrences. |
| Time Above Threshold | Measures time spent above the configured temperature threshold. |
| Thermal Headroom | Measures the difference between the threshold and predicted temperature. |

### Evaluation Objective

The primary evaluation objective is to investigate whether predictive thermal information can improve task offloading decisions compared with reactive and non-predictive approaches.

Results should be reported using repeated simulation runs and consistent workload conditions.

## 🌍 Real-World Applications

The proposed framework has potential applications in environments where computational workloads, thermal conditions, and execution deadlines must be considered together.

### 1. Industrial IoT and Smart Manufacturing

Industrial IoT devices process sensor information and machine-monitoring data. Predictive offloading can help distribute computational workloads when local devices approach thermal limits.

### 2. Smart Healthcare Monitoring

Wearable and healthcare monitoring devices process continuous sensor data. Edge computing can support computationally intensive processing while considering device workload and thermal conditions.

### 3. Connected and Autonomous Vehicles

Connected vehicles generate computationally intensive workloads through sensor processing and environmental analysis. Thermal-aware offloading can support suitable workload distribution in edge-assisted environments.

### 4. Smart City Infrastructure

Smart-city systems process traffic camera feeds, environmental sensor data, and monitoring information. Edge and cloud resources can be considered for distributing these computational tasks.

### 5. Edge Computing Infrastructure

Edge computing environments manage multiple workloads across distributed computing resources. Thermal-aware scheduling and offloading can help account for system temperature and workload conditions.

**Application Scope:** These are potential application areas. The current implementation is a software simulation and does not represent deployment in live industrial, medical, or vehicle systems.

## 💻 Installation and Setup

### Prerequisites

- Python 3.10 or compatible version
- Visual Studio Code or another Python IDE
- Git
- Basic knowledge of Python and machine learning

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Predictive-Thermal-Aware-MEC
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Execution Workflow

Execute the project modules in the following order.

### Step 1: Data Preprocessing

```bash
python data/data_preprocessing.py
```

Prepares the dataset for GRU model training.

### Step 2: Train the GRU Model

```bash
python models/train_gru.py
```

Trains the GRU model to predict future CPU temperature.

### Step 3: Evaluate GRU Predictions

```bash
python evaluation/evaluate_gru.py
```

Evaluates prediction performance using MAE and RMSE.

### Step 4: Visualize GRU Evaluation

```bash
python evaluation/visualize_gru_evaluation.py
```

Generates visualizations for temperature prediction performance.

### Step 5: Initialize the MEC Environment

```bash
python environment/mec_environment.py
```

Initializes the simulated MEC environment.

### Step 6: Train the DQN Agent

```bash
python agents/train_dqn.py
```

Trains the reinforcement learning agent to make task offloading decisions.

### Step 7: Compare DQN Performance

```bash
python evaluation/compare_dqn.py
```

Evaluates the DQN-based offloading strategy.

### Step 8: Execute Baseline Comparison

```bash
python environment/baseline_comparison.py
```

Compares the baseline task offloading strategies.

### Step 9: Visualize Performance Results

```bash
python evaluation/visualize_results.py
```

Generates performance comparison charts.

### Step 10: Launch the Dashboard

```bash
streamlit run app.py
```

Starts the interactive Streamlit application.


### Output
<img width="1870" height="777" alt="image" src="https://github.com/user-attachments/assets/9b6f8b6a-9369-400a-a1e5-c9fadac25d7f" />
<img width="1795" height="463" alt="image" src="https://github.com/user-attachments/assets/39a3898f-9e4d-4394-9386-cf51718bd88d" />
<img width="1872" height="527" alt="image" src="https://github.com/user-attachments/assets/0a05bfeb-5a38-4d84-87d2-91663c12da80" />
<img width="1816" height="831" alt="image" src="https://github.com/user-attachments/assets/117c8fb8-182d-4332-835a-5edfc2e2e2e1" />
<img width="1900" height="692" alt="image" src="https://github.com/user-attachments/assets/bfe49210-7bbf-4638-8784-a9d9d583dde3" />
<img width="1831" height="695" alt="image" src="https://github.com/user-attachments/assets/e5f61426-73fa-4f49-b59d-9cfb2a957ac0" />
<img width="1742" height="635" alt="image" src="https://github.com/user-attachments/assets/53db6447-9118-4fea-af71-709539c56681" />
<img width="1661" height="717" alt="image" src="https://github.com/user-attachments/assets/6ebb7d4a-1989-434f-bfde-f84c8e21c665" />
<img width="1697" height="347" alt="image" src="https://github.com/user-attachments/assets/2587b0bc-7cef-499f-ae9b-b0f0302c965c" />
<img width="1687" height="620" alt="image" src="https://github.com/user-attachments/assets/3d716f04-c5b3-4093-b390-b47e11112d97" />
<img width="1715" height="107" alt="image" src="https://github.com/user-attachments/assets/779018a4-9461-429f-84f7-e0cd7ab0ce19" />

## 👨‍💻 Contributors

**Project Title:** Predictive Thermal-Aware Task Offloading for Deadline-Constrained IoT Applications in Mobile Edge Computing

**Project Domain:** Internet of Things | Mobile Edge Computing | Deep Learning | Reinforcement Learning

**Developed using:** Python, PyTorch, GRU, DQN, and Streamlit.


## 📜 License

This project is intended for academic and research purposes.

---

<p align="center">
  <strong>Predict the Heat. Optimize the Decision. Meet the Deadline.</strong>
</p>

<p align="center">
  Predictive Thermal-Aware Intelligence for Next-Generation Mobile Edge Computing
</p>
