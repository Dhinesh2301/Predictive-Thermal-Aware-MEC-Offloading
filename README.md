# 🌡️ Predictive Thermal-Aware Task Offloading for Deadline-Constrained IoT Applications in Mobile Edge Computing

An **AI-powered Mobile Edge Computing (MEC) task offloading system** that predicts future CPU temperature using a **Gated Recurrent Unit (GRU)** and intelligently selects the optimal task execution location using **Deep Q-Learning (DQN)**.

The system is designed for **deadline-constrained IoT applications**, where decisions must consider not only latency and energy consumption but also the predicted thermal condition of the IoT device.

---

## 📌 Problem Statement

IoT devices often have limited computational resources and may experience:

- High CPU temperature
- Thermal overload
- Increased energy consumption
- High task execution latency
- Missed application deadlines
- Poor edge server selection

Traditional task offloading approaches usually make decisions using only the **current system state**.

However, thermal problems can be predicted before they occur.

This project introduces a **Predictive Thermal-Aware Task Offloading System** that combines:

- 🧠 GRU for future CPU temperature prediction
- 🌡️ Thermal headroom analysis
- 🤖 Deep Q-Learning for intelligent task offloading
- ⚡ MEC-based execution
- ⏱️ Deadline-aware decision making
- 🔋 Energy-aware optimization
- 📊 Baseline performance comparison

The DQN agent selects the most suitable execution target from:

- 💻 LOCAL Device
- 🖥️ EDGE 1
- 🖥️ EDGE 2
- ☁️ CLOUD

---

# 🎯 Project Objectives

The main objectives of this project are:

- Predict future CPU temperature using GRU
- Detect potential thermal risk before task execution
- Calculate thermal headroom dynamically
- Select an optimal task execution location
- Reduce latency and energy consumption
- Satisfy application deadline constraints
- Avoid thermal overload
- Compare intelligent DQN decisions with baseline methods

---

# 🧠 Key Features

- 🌡️ GRU-based CPU temperature prediction
- 🤖 Deep Q-Learning (DQN) task offloading
- 🔥 Thermal-aware decision making
- ⏱️ Deadline-constrained optimization
- ⚡ Latency-aware execution
- 🔋 Energy-aware reward optimization
- 🖥️ Multiple MEC execution targets
- 📊 DQN Q-value analysis
- 📈 GRU model performance evaluation
- 📉 Actual vs predicted temperature visualization
- 🔍 Prediction error analysis
- 📊 Baseline comparison
- 🌐 Interactive Streamlit dashboard
- 💻 Open-source implementation

---

# 🧠 System Architecture
<img width="1536" height="1024" alt="system_architecture" src="https://github.com/user-attachments/assets/64217975-4d35-48c2-9e20-d5a4e73e990c" />
