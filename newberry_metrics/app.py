import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Optional
# from pandas.io.formats.style import Styler # No longer explicitly needed if using st.dataframe
import json
from pathlib import Path
import os
# import argparse # No longer needed for this approach
import glob # For finding session files
# import sys # No longer needed for this approach

# Get the directory of the current script
# This is important if app.py is not in the same CWD as main.py,
# but for session files, we'll assume CWD or allow configuration.
script_dir = Path(__file__).parent.resolve()

# Image path (assuming it's packaged with app.py)
image_file_name = "logo.png"
page_icon_path = script_dir / image_file_name

# --- Configuration for finding session files ---
# Dashboard will look for session files in the current working directory
# This should be the directory from which main.py is executed.
SESSION_FILE_PATTERN = "session_metrics_*.json"
# ---

# Page configuration
st.set_page_config(
    page_title="Newberry Session Metrics",
    page_icon=str(page_icon_path) if page_icon_path.exists() else "📊", # Fallback icon
    layout="wide",
)

# Light theme style settings (can be kept or adjusted)
style = {
    "plotly_template": "plotly_white",
    "line_color": "#6C5CE7",
    "marker_color": "#FDCB6E",
    "bg_color": "#FAFAFA",
    "sidebar_color": "#F0F2F6",
    "text_color": "#2C3E50",
    "chart_bgcolor": "#FFFFFF"
}

# Apply custom light theme via CSS (can be kept or adjusted)
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {style['bg_color']};
            color: {style['text_color']};
        }}
        .css-1d391kg, .css-1v3fvcr, .css-hxt7ib {{ /* Sidebar styling */
            background-color: {style['sidebar_color']} !important;
        }}
        h1, h2, h3, h4, h5, h6, p, div {{ /* General text color */
            color: {style['text_color']} !important;
        }}
        .stDataFrame, .stTable {{ /* Table background */
            background-color: {style['chart_bgcolor']} !important;
        }}
        /* Button styling can be kept or adjusted */
        div.stButton > button {{
            background-color: #6C5CE7 !important;
            color: white !important;
            border-radius: 8px !important;
            border: 1px solid #B2A4FF !important;
            padding: 0.5em 1.5em !important;
            font-weight: bold !important;
            font-size: 1.1em !important;
            margin-bottom: 1em;
            box-shadow: none !important;
            outline: none !important;
        }}
        div.stButton > button:hover, div.stButton > button:active, div.stButton > button:focus {{
             background-color: #584AB7 !important; /* Darker shade on hover/active */
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

def find_latest_session_file(pattern: str) -> Optional[Path]:
    """Finds the most recently modified file matching the pattern in CWD."""
    try:
        session_files = glob.glob(pattern)
        if not session_files:
            return None
        latest_file = max(session_files, key=os.path.getmtime)
        return Path(latest_file)
    except Exception as e:
        st.error(f"Error finding session files: {e}")
        return None

def load_session_data(file_path: Path) -> Optional[dict]:
    """Loads session data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {file_path}. The file might be corrupted or empty.")
        return None
    except IOError:
        st.error(f"Error: Could not read session file: {file_path}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading {file_path}: {e}")
        return None

# --- Load Data ---
latest_session_file_path = find_latest_session_file(SESSION_FILE_PATTERN)
session_data = None
df_api_calls = pd.DataFrame() # Initialize empty DataFrame

if latest_session_file_path:
    st.info(f"Loading data from: {latest_session_file_path.name}")
    session_data = load_session_data(latest_session_file_path)
    if session_data and "api_calls" in session_data:
        if session_data["api_calls"]: # Check if there are any calls
            df_api_calls = pd.DataFrame(session_data["api_calls"])
            # Ensure essential columns exist, even if empty
            for col in ['cost', 'latency', 'input_tokens', 'output_tokens', 'call_counter', 'timestamp']:
                if col not in df_api_calls.columns:
                    df_api_calls[col] = 0 if col != 'timestamp' else pd.NaT
            if 'timestamp' in df_api_calls.columns:
                 df_api_calls['timestamp'] = pd.to_datetime(df_api_calls['timestamp'], errors='coerce')
        else: # api_calls list is empty
             st.warning("Session file loaded, but no API calls recorded in this session yet.")
    elif session_data is None: # load_session_data returned None due to an error
        pass # Error message already shown by load_session_data
    else: # session_data loaded but no "api_calls" key or it's None
        st.error("Session file format is unexpected. Missing 'api_calls' list.")
        session_data = None # Ensure session_data is None if format is bad

else:
    st.warning(f"No session data files ('{SESSION_FILE_PATTERN}') found in the current directory.")
    st.markdown("Please run your main application script to generate metrics.")

# --- Page Title & Logo ---
header_col1, header_col2 = st.columns([1, 10])
with header_col1:
    if page_icon_path.exists():
        st.image(str(page_icon_path), width=100) # Adjusted width
with header_col2:
    st.title("Newberry Session Metrics")

# --- KPI Display ---
st.header("Session Overview")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

if session_data:
    total_c = session_data.get("total_cost", 0.0)
    avg_c = session_data.get("average_cost", 0.0)
    total_l = session_data.get("total_latency", 0.0)
    avg_l = session_data.get("average_latency", 0.0)
    total_calls_count = session_data.get("total_calls", 0)

    kpi1.metric(label="Total Cost", value=f"${total_c:.6f}")
    kpi2.metric(label="Avg Cost/Call", value=f"${avg_c:.6f}")
    kpi3.metric(label="Total Latency", value=f"{total_l:.3f} s")
    kpi4.metric(label="Avg Latency/Call", value=f"{avg_l:.3f} s")
    kpi5.metric(label="Total API Calls", value=f"{total_calls_count}")
else:
    kpi1.metric(label="Total Cost", value="$0.00")
    kpi2.metric(label="Avg Cost/Call", value="$0.00")
    kpi3.metric(label="Total Latency", value="0.000 s")
    kpi4.metric(label="Avg Latency/Call", value="0.000 s")
    kpi5.metric(label="Total API Calls", value="0")


# --- Charts ---
st.header("Visualizations")
if not df_api_calls.empty:
    # Line chart: Total Cost Over Session (cumulative cost)
    if 'cost' in df_api_calls.columns:
        df_api_calls['cumulative_cost'] = df_api_calls['cost'].cumsum()
        fig1 = px.line(
            df_api_calls,
            x=df_api_calls.index, # Or 'call_counter' if preferred and 1-indexed
            y='cumulative_cost',
            title="Cumulative Cost Over Session",
            template=style['plotly_template'],
            labels={'cumulative_cost': 'Cumulative Cost ($)', 'x': 'API Call Index'}
        )
        fig1.update_traces(line=dict(color=style['line_color']))
        fig1.update_layout(plot_bgcolor=style['chart_bgcolor'], paper_bgcolor=style['chart_bgcolor'])
        st.plotly_chart(fig1, use_container_width=True)

    # Scatter chart: Cost vs Latency per call
    if 'latency' in df_api_calls.columns and 'cost' in df_api_calls.columns:
        fig2 = px.scatter(
            df_api_calls,
            x='latency',
            y='cost',
            title="Cost vs Latency (Per API Call)",
            template=style['plotly_template'],
            labels={'latency': 'Latency (s)', 'cost': 'Cost ($)'}
        )
        fig2.update_traces(marker=dict(color=style['marker_color'], size=10))
        fig2.update_layout(plot_bgcolor=style['chart_bgcolor'], paper_bgcolor=style['chart_bgcolor'])
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No API call data to display in charts.")

# --- Detailed Data View ---
st.header("Detailed API Call Data")
if not df_api_calls.empty:
    # Define columns to display and their formatting
    display_df = df_api_calls[['timestamp', 'call_counter', 'cost', 'latency', 'input_tokens', 'output_tokens']].copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    st.dataframe(
        display_df.style.format({
            'cost': '{:.6f}',
            'latency': '{:.3f}',
        }),
        use_container_width=True
    )
    # Option to download data
    @st.cache_data # Cache the conversion to CSV
    def convert_df_to_csv(df_to_convert):
        return df_to_convert.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(display_df)
    st.download_button(
        label="Download data as CSV",
        data=csv_data,
        file_name=f"{latest_session_file_path.stem}_apicalls.csv" if latest_session_file_path else "api_calls.csv",
        mime='text/csv',
    )

else:
    st.info("No detailed API call data to display.")


# --- Sidebar with information (Optional: Can be kept or simplified) ---
with st.sidebar:
    if page_icon_path.exists():
        st.image(str(page_icon_path), width=200)
    st.markdown(
        """
        <div style='padding-top: 10px;'>
        <h4 style='color:#6C5CE7;'>Newberry Metrics</h4>
        <p style='color:#2C3E50;'><b>What is it?</b><br>
        A lightweight Python package to track cost, latency, and performance metrics of LLMs on Amazon Bedrock.
        </p>
        <p style='color:#2C3E50;'><b>How it works:</b><br>
        This dashboard displays metrics from the most recent session initiated by the <code>Newberry TokenEstimator</code>.
        </p>
        <p style='color:#2C3E50;'>
        <b>Version:</b> 1.0.5 (Example)
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Refresh Data"):
        st.rerun()
