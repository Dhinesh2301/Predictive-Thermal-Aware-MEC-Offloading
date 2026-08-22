import streamlit as st
import subprocess
import re
import pandas as pd
from datetime import datetime
from pathlib import Path
import os
import sys


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Predictive Thermal-Aware MEC Offloading",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# PROJECT PATH
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

PREDICT_SCRIPT = BASE_DIR / "predict.py"
COMPARE_SCRIPT = BASE_DIR / "evaluation" / "compare_dqn.py"

LATENCY_GRAPH = BASE_DIR / "results" / "latency_comparison.png"
ENERGY_GRAPH = BASE_DIR / "results" / "energy_comparison.png"
REWARD_GRAPH = BASE_DIR / "results" / "reward_comparison.png"



# ==========================================================
# GRU EVALUATION GRAPHS
# ==========================================================

GRU_ACTUAL_PREDICTED_GRAPH = (
    BASE_DIR / "results" / "gru_actual_vs_predicted.png"
)

GRU_ERROR_DISTRIBUTION_GRAPH = (
    BASE_DIR / "results" / "gru_error_distribution.png"
)

GRU_ABSOLUTE_ERROR_GRAPH = (
    BASE_DIR / "results" / "gru_absolute_error.png"
)

GRU_ACCURACY_GRAPH = (
    BASE_DIR / "results" / "gru_prediction_accuracy.png"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.dashboard-title {
    text-align: center;
    font-size: 46px;
    font-weight: bold;
    color: #00d4ff;
    margin-bottom: 5px;
}

.dashboard-subtitle {
    text-align: center;
    font-size: 19px;
    color: #9aa4b2;
    margin-bottom: 25px;
}

.section-title {
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
    color: #f5f5f5;
}

.pipeline-box {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 10px;
    text-align: center;
}

.pipeline-number {
    font-size: 14px;
    color: #00d4ff;
    font-weight: bold;
}

.pipeline-name {
    font-size: 16px;
    font-weight: bold;
    color: white;
}

.status-box {
    background-color: #161b22;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #30363d;
}

.reward-card {
    background-color: #161b22;
    border: 1px solid #00d4ff;
    border-radius: 12px;
    padding: 18px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# SESSION STATE
# ==========================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "analysis_completed" not in st.session_state:
    st.session_state.analysis_completed = False

if "temperature" not in st.session_state:
    st.session_state.temperature = None

if "thermal_status" not in st.session_state:
    st.session_state.thermal_status = None

if "comparison_output" not in st.session_state:
    st.session_state.comparison_output = ""

if "comparison_data" not in st.session_state:
    st.session_state.comparison_data = None


# ==========================================================
# HELPER FUNCTION - RUN PYTHON FILE
# ==========================================================

def run_python_script(script_path, arguments=None):

    try:

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        command = [
            sys.executable,
            str(script_path)
        ]

        if arguments:

            command.extend(
                [str(arg) for arg in arguments]
            )

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            encoding="utf-8",
            errors="replace",
            env=env
        )

        return result.stdout, result.stderr

    except Exception as e:

        return "", str(e)


# ==========================================================
# HELPER FUNCTION - EXTRACT VALUE
# ==========================================================

def extract_value(pattern, text):

    match = re.search(
        pattern,
        text
    )

    if match:
        return match.group(1)

    return None


# ==========================================================
# EXTRACT COMPARISON RESULTS
# ==========================================================

def parse_comparison_output(output):

    data = {}

    # ------------------------------------------------------
    # SYSTEM STATE
    # ------------------------------------------------------

    predicted_temp = extract_value(
        r"Predicted Temperature\s*:\s*([\d.]+)",
        output
    )

    thermal_headroom = extract_value(
        r"Thermal Headroom\s*:\s*([\d.]+)",
        output
    )

    task_cycles = extract_value(
        r"Task CPU Cycles\s*:\s*([\d.]+)",
        output
    )

    task_input = extract_value(
        r"Task Input Size\s*:\s*([\d.]+)",
        output
    )

    deadline = extract_value(
        r"Task Deadline\s*:\s*([\d.]+)",
        output
    )

    edge1_load = extract_value(
        r"Edge 1 Load\s*:\s*([\d.]+)",
        output
    )

    edge2_load = extract_value(
        r"Edge 2 Load\s*:\s*([\d.]+)",
        output
    )


    # ------------------------------------------------------
    # DQN FINAL DECISION
    # ------------------------------------------------------

    dqn_action = extract_value(
        r"DQN Selected Action\s*:\s*(\d+)",
        output
    )

    dqn_location = extract_value(
        r"Execution Location\s*:\s*(LOCAL|EDGE\s*1|EDGE\s*2|CLOUD)",
        output
    )

    dqn_section = output.split(
        "TRAINED DQN DECISION"
    )[-1]

    dqn_latency = extract_value(
        r"Latency\s*:\s*([\d.]+)\s*ms",
        dqn_section
    )

    dqn_energy = extract_value(
        r"Energy\s*:\s*([\d.]+)\s*J",
        dqn_section
    )

    dqn_reward = extract_value(
        r"DQN Reward\s*:\s*(-?[\d.]+)",
        output
    )


    # ------------------------------------------------------
    # BEST BASELINE
    # ------------------------------------------------------

    best_baseline = extract_value(
        r"Best Baseline\s*:\s*(LOCAL|EDGE\s*1|EDGE\s*2|CLOUD)",
        output
    )

    baseline_reward = extract_value(
        r"Baseline Reward\s*:\s*(-?[\d.]+)",
        output
    )


    # ------------------------------------------------------
    # Q VALUES
    # ------------------------------------------------------

    local_q = extract_value(
        r"LOCAL\s*→\s*(-?[\d.]+)",
        output
    )

    edge1_q = extract_value(
        r"EDGE 1\s*→\s*(-?[\d.]+)",
        output
    )

    edge2_q = extract_value(
        r"EDGE 2\s*→\s*(-?[\d.]+)",
        output
    )

    cloud_q = extract_value(
        r"CLOUD\s*→\s*(-?[\d.]+)",
        output
    )


    # ------------------------------------------------------
    # STORE VALUES
    # ------------------------------------------------------

    data["temperature"] = (
        float(predicted_temp)
        if predicted_temp
        else 0
    )

    data["headroom"] = (
        float(thermal_headroom)
        if thermal_headroom
        else 0
    )

    data["task_cycles"] = (
        float(task_cycles)
        if task_cycles
        else 0
    )

    data["task_input"] = (
        float(task_input)
        if task_input
        else 0
    )

    data["deadline"] = (
        float(deadline)
        if deadline
        else 0
    )

    data["edge1_load"] = (
        float(edge1_load)
        if edge1_load
        else 0
    )

    data["edge2_load"] = (
        float(edge2_load)
        if edge2_load
        else 0
    )

    data["dqn_action"] = (
        dqn_action
        if dqn_action
        else "N/A"
    )

    data["dqn_location"] = (
        dqn_location
        if dqn_location
        else "N/A"
    )

    data["dqn_latency"] = (
        float(dqn_latency)
        if dqn_latency
        else 0
    )

    data["dqn_energy"] = (
        float(dqn_energy)
        if dqn_energy
        else 0
    )

    data["dqn_reward"] = (
        float(dqn_reward)
        if dqn_reward
        else 0
    )

    data["best_baseline"] = (
        best_baseline
        if best_baseline
        else "N/A"
    )

    data["baseline_reward"] = (
        float(baseline_reward)
        if baseline_reward
        else 0
    )

    data["q_values"] = {
        "LOCAL": float(local_q) if local_q else 0,
        "EDGE 1": float(edge1_q) if edge1_q else 0,
        "EDGE 2": float(edge2_q) if edge2_q else 0,
        "CLOUD": float(cloud_q) if cloud_q else 0
    }

    return data


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("🌡️ MEC Control Panel")

    st.success("🟢 System Online")

    st.markdown("---")

    st.subheader("AI Components")

    st.write("🧠 GRU Temperature Predictor")
    st.write("🤖 DQN Offloading Agent")
    st.write("🌡️ Thermal Manager")
    st.write("⚡ Reward System")

    st.markdown("---")

    st.subheader("Available Actions")

    st.write("0 → LOCAL")
    st.write("1 → EDGE 1")
    st.write("2 → EDGE 2")
    st.write("3 → CLOUD")

    st.markdown("---")

    st.caption(
        "Predictive Thermal-Aware MEC "
        "Offloading System"
    )


# ==========================================================
# TITLE
# ==========================================================

st.markdown(
    """
    <div class="dashboard-title">
        🌡️ Predictive Thermal-Aware MEC Offloading
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="dashboard-subtitle">
        AI-Powered GRU Temperature Prediction + DQN Intelligent Offloading
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# SYSTEM STATUS
# ==========================================================

st.markdown(
    '<div class="section-title">🖥️ System Monitoring</div>',
    unsafe_allow_html=True
)

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

status_col1.metric(
    "AI Model",
    "GRU Active",
    "🟢 Ready"
)

status_col2.metric(
    "Decision Agent",
    "DQN Active",
    "🤖 Learning"
)

status_col3.metric(
    "Prediction Engine",
    "Online",
    "⚡ Fast"
)

status_col4.metric(
    "Thermal Monitoring",
    "Active",
    "🌡️ Live"
)


st.divider()


# ==========================================================
# PROJECT PIPELINE
# ==========================================================

st.markdown(
    '<div class="section-title">🔄 Intelligent MEC Decision Pipeline</div>',
    unsafe_allow_html=True
)

pipeline = [
    "📊 Dataset",
    "🧹 Preprocessing",
    "🧠 GRU Prediction",
    "🌡️ Headroom",
    "🤖 DQN State",
    "🎯 DQN Action",
    "⚡ MEC Execution",
    "💰 Reward"
]

pipeline_cols = st.columns(8)

for i, step in enumerate(pipeline):

    pipeline_cols[i].markdown(
        f"""
        <div class="pipeline-box">
            <div class="pipeline-number">STEP {i + 1}</div>
            <div class="pipeline-name">{step}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# RUN COMPLETE SYSTEM
# ==========================================================

st.markdown(
    '<div class="section-title">🚀 Run Intelligent MEC Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "The system will predict future CPU temperature, "
    "calculate thermal headroom, allow the trained DQN "
    "agent to select an offloading action, and calculate "
    "reward."
)


if st.button(
    "🚀 Run GRU + DQN Intelligent Decision",
    use_container_width=True
):

    # ------------------------------------------------------
    # STEP 1 - RUN GRU PREDICTION
    # ------------------------------------------------------

    with st.spinner(
        "🧠 GRU is predicting future CPU temperature..."
    ):

        prediction_output, prediction_error = (
            run_python_script(
                PREDICT_SCRIPT
            )
        )


    if prediction_error:

        st.error(
            "Prediction error occurred."
        )

        st.code(
            prediction_error
        )


    else:

        temperature_match = re.search(
            r"Predicted CPU Temperature:\s*([\d.]+)",
            prediction_output
        )

        status_match = re.search(
            r"Thermal Status:\s*(\w+)",
            prediction_output
        )


        if temperature_match:

            temperature = float(
                temperature_match.group(1)
            )

            if status_match:

                thermal_status = (
                    status_match.group(1)
                )

            else:

                thermal_status = "UNKNOWN"


            st.session_state.temperature = (
                temperature
            )

            st.session_state.thermal_status = (
                thermal_status
            )


            current_time = datetime.now().strftime(
                "%H:%M:%S"
            )


            st.session_state.prediction_history.append(
                {
                    "Time": current_time,
                    "Predicted Temperature": temperature
                }
            )


            if len(
                st.session_state.prediction_history
            ) > 20:

                st.session_state.prediction_history.pop(0)


            # --------------------------------------------------
            # STEP 2 - RUN DQN COMPARISON
            #
            # IMPORTANT:
            # Send GRU predicted temperature to compare_dqn.py
            # --------------------------------------------------

            with st.spinner(
                "🤖 Trained DQN agent is selecting optimal MEC action..."
            ):

                comparison_output, comparison_error = (
                    run_python_script(
                        COMPARE_SCRIPT,
                        arguments=[temperature]
                    )
                )


            if comparison_error:

                st.warning(
                    "DQN comparison produced an error."
                )

                st.code(
                    comparison_error
                )


            # --------------------------------------------------
            # PARSE DQN OUTPUT
            # --------------------------------------------------

            comparison_data = parse_comparison_output(
                comparison_output
            )


            # --------------------------------------------------
            # IMPORTANT:
            # Force comparison temperature to use the
            # SAME GRU predicted temperature.
            # --------------------------------------------------

            comparison_data["temperature"] = (
                temperature
            )


            # Calculate correct thermal headroom
            comparison_data["headroom"] = (
                85 - temperature
            )


            st.session_state.comparison_output = (
                comparison_output
            )

            st.session_state.comparison_data = (
                comparison_data
            )

            st.session_state.analysis_completed = True


            st.success(
                "🎉 Complete GRU + DQN intelligent analysis completed successfully!"
            )


        else:

            st.error(
                "Unable to extract GRU temperature prediction."
            )

            st.code(
                prediction_output
            )


# ==========================================================
# RESULTS DASHBOARD
# ==========================================================

if st.session_state.analysis_completed:

    data = st.session_state.comparison_data

    # ======================================================
    # IMPORTANT:
    # Always use the latest GRU predicted temperature
    # ======================================================

    temperature = st.session_state.temperature


    # ======================================================
    # THERMAL ANALYSIS
    # ======================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🌡️ GRU Thermal Prediction Analysis</div>',
        unsafe_allow_html=True
    )

    thermal_col1, thermal_col2, thermal_col3, thermal_col4 = (
        st.columns(4)
    )


    thermal_col1.metric(
        "Predicted CPU Temperature",
        f"{temperature:.2f} °C"
    )


    thermal_col2.metric(
        "Thermal Status",
        st.session_state.thermal_status
    )


    thermal_col3.metric(
        "Thermal Headroom",
        f"{data['headroom']:.2f} °C"
    )


    thermal_percentage = min(
        temperature,
        100
    )


    thermal_col4.metric(
        "Thermal Level",
        f"{thermal_percentage:.1f}%"
    )


    st.progress(
        int(thermal_percentage)
    )


    if temperature < 50:

        st.success(
            "🟢 COOL: System is operating safely."
        )

    elif temperature < 70:

        st.info(
            "🔵 NORMAL: System temperature is stable."
        )

    elif temperature < 85:

        st.warning(
            "🟠 HIGH: CPU temperature is high. "
            "Thermal-aware offloading is recommended."
        )

    else:

        st.error(
            "🔴 HOT: CPU temperature is critically high!"
        )


    # ======================================================
    # DQN STATE
    # ======================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🤖 DQN Agent System State</div>',
        unsafe_allow_html=True
    )

    state_col1, state_col2, state_col3, state_col4 = (
        st.columns(4)
    )


    state_col1.metric(
        "Task CPU Cycles",
        f"{data['task_cycles']:.0f} M"
    )


    state_col2.metric(
        "Task Input Size",
        f"{data['task_input']:.2f} MB"
    )


    state_col3.metric(
        "Task Deadline",
        f"{data['deadline']:.2f} ms"
    )


    state_col4.metric(
        "Thermal Headroom",
        f"{data['headroom']:.2f} °C"
    )


    load_col1, load_col2 = st.columns(2)


    load_col1.metric(
        "Edge 1 Server Load",
        f"{data['edge1_load']:.1f}%"
    )


    load_col2.metric(
        "Edge 2 Server Load",
        f"{data['edge2_load']:.1f}%"
    )


    # ======================================================
    # DQN OFFLOADING DECISION
    # ======================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🎯 DQN Intelligent Offloading Decision</div>',
        unsafe_allow_html=True
    )

    decision_col1, decision_col2, decision_col3, decision_col4 = (
        st.columns(4)
    )


    decision_col1.metric(
        "Selected Action",
        data["dqn_action"]
    )


    decision_col2.metric(
        "Execution Target",
        data["dqn_location"]
    )


    decision_col3.metric(
        "Expected Latency",
        f"{data['dqn_latency']:.2f} ms"
    )


    decision_col4.metric(
        "Energy Consumption",
        f"{data['dqn_energy']:.2f} J"
    )


    # ======================================================
    # DQN REWARD
    # ======================================================

    st.markdown(
        '<div class="section-title">💰 DQN Reward Evaluation</div>',
        unsafe_allow_html=True
    )

    reward_col1, reward_col2, reward_col3 = (
        st.columns(3)
    )


    reward_col1.metric(
        "Final DQN Reward",
        f"{data['dqn_reward']:.2f}"
    )


    reward_col2.metric(
        "Baseline Reward",
        f"{data['baseline_reward']:.2f}"
    )


    reward_difference = (
        data["dqn_reward"]
        -
        data["baseline_reward"]
    )


    reward_col3.metric(
        "Reward Difference",
        f"{reward_difference:.2f}"
    )


    if data["dqn_reward"] > 0:

        st.success(
            "🏆 Positive reward: The selected MEC decision "
            "satisfies the system optimization objective."
        )

    else:

        st.error(
            "⚠️ Negative reward: The selected action violates "
            "important system constraints."
        )


    # ======================================================
    # Q VALUE ANALYSIS
    # ======================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🧠 DQN Q-Value Analysis</div>',
        unsafe_allow_html=True
    )

    q_df = pd.DataFrame(
        {
            "Execution Target": list(
                data["q_values"].keys()
            ),
            "DQN Q-Value": list(
                data["q_values"].values()
            )
        }
    )


    q_col1, q_col2 = st.columns([1, 2])


    with q_col1:

        st.dataframe(
            q_df,
            use_container_width=True,
            hide_index=True
        )


    with q_col2:

        st.bar_chart(
            q_df.set_index(
                "Execution Target"
            )
        )


    best_q_action = max(
        data["q_values"],
        key=data["q_values"].get
    )


    st.info(
        f"🎯 Highest Q-Value Action: **{best_q_action}** "
        f"with Q-value **{data['q_values'][best_q_action]:.4f}**"
    )


    # ======================================================
    # CONSTRAINT VALIDATION
    # ======================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🛡️ Deadline and Thermal Constraint Validation</div>',
        unsafe_allow_html=True
    )

    constraint_col1, constraint_col2, constraint_col3 = (
        st.columns(3)
    )


    deadline_met = (
        data["dqn_latency"]
        <=
        data["deadline"]
    )


    thermal_safe = (
        data["headroom"] >= 0
    )


    constraint_col1.metric(
        "Deadline Constraint",
        "PASS" if deadline_met else "FAILED"
    )


    constraint_col2.metric(
        "Thermal Constraint",
        "SAFE" if thermal_safe else "VIOLATED"
    )


    constraint_col3.metric(
        "Final Decision",
        data["dqn_location"]
    )


    if deadline_met and thermal_safe:

        st.success(
            "✅ The DQN-selected MEC target satisfies both "
            "deadline and thermal constraints."
        )

    else:

        st.error(
            "❌ The selected action violates one or more constraints."
        )


    # ======================================================
    # BASELINE COMPARISON
    # ======================================================

    st.divider()

    st.markdown(
        '<div class="section-title">⚖️ DQN vs Baseline Performance</div>',
        unsafe_allow_html=True
    )


    comparison_col1, comparison_col2 = st.columns(2)


    comparison_col1.metric(
        "Best Baseline",
        data["best_baseline"]
    )


    comparison_col2.metric(
        "DQN Decision",
        data["dqn_location"]
    )


    if data["dqn_reward"] >= data["baseline_reward"]:

        st.success(
            "🏆 DQN selected an optimal or equal-best MEC offloading decision."
        )

    else:

        st.warning(
            "⚠️ DQN selected a decision below the best baseline reward."
        )


    # ======================================================
    # PERFORMANCE GRAPHS
    # ======================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Performance Evaluation Graphs</div>',
        unsafe_allow_html=True
    )

    graph_col1, graph_col2 = st.columns(2)


    with graph_col1:

        if LATENCY_GRAPH.exists():

            st.image(
                str(LATENCY_GRAPH),
                caption="Latency Comparison",
                use_container_width=True
            )


    with graph_col2:

        if ENERGY_GRAPH.exists():

            st.image(
                str(ENERGY_GRAPH),
                caption="Energy Comparison",
                use_container_width=True
            )


    if REWARD_GRAPH.exists():

        st.image(
            str(REWARD_GRAPH),
            caption="DQN Reward Comparison",
            use_container_width=True
        )


    # ======================================================
    # LIVE PREDICTION HISTORY
    # ======================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📈 Live GRU Prediction History</div>',
        unsafe_allow_html=True
    )


    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )


    if len(history_df) > 0:

        st.line_chart(
            history_df,
            x="Time",
            y="Predicted Temperature",
            use_container_width=True
        )


        history_col1, history_col2, history_col3 = (
            st.columns(3)
        )


        history_col1.metric(
            "Minimum Temperature",
            f"{history_df['Predicted Temperature'].min():.2f} °C"
        )


        history_col2.metric(
            "Average Temperature",
            f"{history_df['Predicted Temperature'].mean():.2f} °C"
        )


        history_col3.metric(
            "Maximum Temperature",
            f"{history_df['Predicted Temperature'].max():.2f} °C"
        )


        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )


    # ======================================================
    # CLEAR HISTORY
    # ======================================================

    if st.button(
        "🗑️ Clear Prediction History"
    ):

        st.session_state.prediction_history = []

        st.rerun()


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "🌡️ Predictive Thermal-Aware MEC Offloading System | "
    "📊 Computer Metrics Dataset | "
    "🧠 GRU Temperature Prediction | "
    "🤖 DQN Intelligent Offloading | "
    "💰 Thermal-Aware Reward Optimization"
)