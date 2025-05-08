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
    st.image("/Users/harshikaagarwal/Desktop/newberry/newberry_metrics/newberry_metrics/Screenshot_2025-05-06_at_4.25.24_PM-removebg-preview (1).png", width=250)
with header_col2:
    st.title("Newberry Session Metrics")

# KPI calculations
avg_cost = df['cost'].mean()
total_cost = df['cost'].sum()
avg_latency = df['latency'].mean()
total_latency = df['latency'].sum()

# KPI display
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(label="Avg Cost", value=f"${avg_cost:.6f}")
kpi2.metric(label="Total Cost", value=f"${total_cost:.6f}")
kpi3.metric(label="Avg Latency", value=f"{avg_latency:.6f} ms")
kpi4.metric(label="Total Latency", value=f"{total_latency:.6f} ms")

# Add dropdown menu for view selection
view_options = ["Hourly View", "Daily View"]
selected_view = st.selectbox(
    "Select View",
    options=view_options,
    key="view_selector"
)

# Style the selectbox to match the theme
st.markdown(
    """
    <style>
        div[data-baseweb="select"] {
            background-color: #FFFFFF !important;
            border-radius: 8px !important;
            border: 1px solid #B2A4FF !important;
        }
        div[data-baseweb="select"] > div {
            color: #2C3E50 !important;
            font-weight: bold !important;
            background-color: #FFFFFF !important;
        }
        div[data-baseweb="select"] > div[aria-selected="true"] {
            background-color: #FFFFFF !important;
            color: #6C5CE7 !important;
        }
        div[data-baseweb="select"] > div:hover {
            background-color: #F0F2F6 !important;
        }
        /* Force light background for dropdown options */
        div[data-baseweb="popover"] {
            background-color: #FFFFFF !important;
            border: 1px solid #B2A4FF !important;
            border-radius: 8px !important;
        }
        div[data-baseweb="popover"] * {
            background-color: #FFFFFF !important;
            color: #2C3E50 !important;
        }
        div[data-baseweb="popover"] [role="option"] {
            background-color: #FFFFFF !important;
            color: #2C3E50 !important;
        }
        div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
            background-color: #F0F2F6 !important;
            color: #6C5CE7 !important;
        }
        div[data-baseweb="popover"] [role="option"]:hover {
            background-color: #F0F2F6 !important;
            color: #6C5CE7 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- CHARTS BASED ON DROPDOWN SELECTION ---
if selected_view == "Hourly View":
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_hour = df.copy()
    df_hour['hour'] = df_hour['timestamp'].dt.strftime('%Y-%m-%d %H:00')
    hourly_cost = df_hour.groupby('hour')['cost'].sum().reset_index()
    hourly_latency = df_hour.groupby('hour')['latency'].mean().reset_index()
    
    # Calculate hourly input-output ratio
    hourly_io = df_hour.groupby('hour').agg({
        'input_tokens': 'sum',
        'output_tokens': 'sum'
    }).reset_index()
    hourly_io['io_ratio'] = hourly_io['output_tokens'] / hourly_io['input_tokens']

    # Create input-output ratio bar chart
    fig3 = px.bar(
        hourly_io,
        x='hour',
        y=['input_tokens', 'output_tokens'],
        title="<i>Hourly Input-Output Token Distribution</i>",
        template=style['plotly_template'],
        barmode='group',
        color_discrete_sequence=['#87CEEB', '#FFE5B4']  # Light blue and light yellow
    )
    fig3.update_layout(
        plot_bgcolor=style['chart_bgcolor'],
        paper_bgcolor=style['chart_bgcolor'],
        font=dict(family='Montserrat, Poppins, Segoe UI, Arial', color='#6C5CE7', size=14),
        title_font=dict(family='Montserrat, Poppins, Segoe UI, Arial', size=20, color='#6C5CE7'),
        title={"text": "<i>Hourly Input-Output Token Distribution</i>", "font": {"color": "#6C5CE7"}},
        xaxis=dict(
            title='Hour',
            title_font=dict(color='#87CEEB'),
            tickfont=dict(color='#87CEEB')
        ),
        yaxis=dict(
            title='Number of Tokens',
            title_font=dict(color='#87CEEB'),
            tickfont=dict(color='#87CEEB')
        ),
        legend=dict(
            title='Token Type',
            title_font=dict(color='#87CEEB'),
            font=dict(color='#87CEEB')
        )
    )
    st.plotly_chart(fig3, use_container_width=True)

    fig1 = px.line(
        hourly_cost,
        x='hour',
        y='cost',
        title="<i>Hourly Cost</i>",
        template=style['plotly_template'],
    )
    fig1.update_traces(line=dict(color=style['line_color']))
    fig1.update_layout(
        plot_bgcolor=style['chart_bgcolor'],
        paper_bgcolor=style['chart_bgcolor'],
        font=dict(family='Montserrat, Poppins, Segoe UI, Arial', color='#6C5CE7', size=14),
        title_font=dict(family='Montserrat, Poppins, Segoe UI, Arial', size=20, color='#6C5CE7'),
        title={"text": "<i>Hourly Cost</i>", "font": {"color": "#6C5CE7"}},
        xaxis=dict(
            title='Hour',
            title_font=dict(color='#87CEEB'),
            tickfont=dict(color='#87CEEB')
        ),
        yaxis=dict(
            title='Cost ($)',
            title_font=dict(color='#87CEEB'),
            tickfont=dict(color='#87CEEB'),
            tickformat='$,.6f'
        )
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        hourly_latency,
        x='hour',
        y='latency',
        title="<i>Hourly Latency</i>",
        template=style['plotly_template'],
    )
    fig2.update_traces(marker=dict(color=style['marker_color']))
    fig2.update_layout(
        plot_bgcolor=style['chart_bgcolor'],
        paper_bgcolor=style['chart_bgcolor'],
        font=dict(family='Montserrat, Poppins, Segoe UI, Arial', color='#6C5CE7', size=14),
        title_font=dict(family='Montserrat, Poppins, Segoe UI, Arial', size=20, color='#6C5CE7'),
        title={"text": "<i>Hourly Latency</i>", "font": {"color": "#6C5CE7"}},
        xaxis=dict(
            title='Hour',
            title_font=dict(color='#87CEEB'),
            tickfont=dict(color='#87CEEB')
        ),
        yaxis=dict(
            title='Latency (ms)',
            title_font=dict(color='#87CEEB'),
            tickfont=dict(color='#87CEEB')
        )
    )
    st.plotly_chart(fig2, use_container_width=True)

elif selected_view == "Daily View":
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_day = df.copy()
    df_day['day'] = df_day['timestamp'].dt.strftime('%Y-%m-%d')
    daily_cost = df_day.groupby('day')['cost'].sum().reset_index()
    daily_latency = df_day.groupby('day')['latency'].mean().reset_index()
    
    # Calculate daily input-output ratio
    daily_io = df_day.groupby('day').agg({
        'input_tokens': 'sum',
        'output_tokens': 'sum'
    }).reset_index()
    daily_io['io_ratio'] = daily_io['output_tokens'] / daily_io['input_tokens']

    # Create input-output ratio bar chart
    fig3 = px.bar(
        daily_io,
        x='day',
        y=['input_tokens', 'output_tokens'],
        title="<i>Daily Input-Output Token Distribution</i>",
        template=style['plotly_template'],
        barmode='group',
        color_discrete_sequence=['#87CEEB', '#FFE5B4']  # Light blue and light yellow
    )
    fig3.update_layout(
        plot_bgcolor=style['chart_bgcolor'],
        paper_bgcolor=style['chart_bgcolor'],
        font=dict(family='Montserrat, Poppins, Segoe UI, Arial', color='#6C5CE7', size=14),
        title_font=dict(family='Montserrat, Poppins, Segoe UI, Arial', size=20, color='#6C5CE7'),
        title={"text": "<i>Daily Input-Output Token Distribution</i>", "font": {"color": "#6C5CE7"}},
        xaxis=dict(
            title='Day',
            title_font=dict(color='#87CEEB'),
            tickfont=dict(color='#87CEEB')
        ),
        yaxis=dict(
            title='Number of Tokens',
            title_font=dict(color='#87CEEB'),
            tickfont=dict(color='#87CEEB')
        ),
        legend=dict(
            title='Token Type',
            title_font=dict(color='#87CEEB'),
            font=dict(color='#87CEEB')
        )
    )
    st.plotly_chart(fig3, use_container_width=True)

    fig1 = px.line(
        daily_cost,
        x='day',
        y='cost',
        title="<i>Daily Cost</i>",
        template=style['plotly_template'],
    )
    fig1.update_traces(line=dict(color=style['line_color']))
    fig1.update_layout(
        plot_bgcolor=style['chart_bgcolor'],
        paper_bgcolor=style['chart_bgcolor'],
        font=dict(family='Montserrat, Poppins, Segoe UI, Arial', color='#6C5CE7', size=14),
        title_font=dict(family='Montserrat, Poppins, Segoe UI, Arial', size=20, color='#6C5CE7'),
        title={"text": "<i>Daily Cost</i>", "font": {"color": "#6C5CE7"}},
        xaxis=dict(
            title='Day',
            title_font=dict(color='#87CEEB'),
            tickfont=dict(color='#87CEEB')
        ),
        yaxis=dict(
            title='Cost ($)',
            title_font=dict(color='#87CEEB'),
            tickfont=dict(color='#87CEEB'),
            tickformat='$,.6f'
        )
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        daily_latency,
        x='day',
        y='latency',
        title="<i>Daily Latency</i>",
        template=style['plotly_template'],
    )
    fig2.update_traces(marker=dict(color=style['marker_color']))
    fig2.update_layout(
        plot_bgcolor=style['chart_bgcolor'],
        paper_bgcolor=style['chart_bgcolor'],
        font=dict(family='Montserrat, Poppins, Segoe UI, Arial', color='#6C5CE7', size=14),
        title_font=dict(family='Montserrat, Poppins, Segoe UI, Arial', size=20, color='#6C5CE7'),
        title={"text": "<i>Daily Latency</i>", "font": {"color": "#6C5CE7"}},
        xaxis=dict(
            title='Day',
            title_font=dict(color='#87CEEB'),
            tickfont=dict(color='#87CEEB')
        ),
        yaxis=dict(
            title='Latency (ms)',
            title_font=dict(color='#87CEEB'),
            tickfont=dict(color='#87CEEB')
        )
    )
    st.plotly_chart(fig2, use_container_width=True)

# Styled DataFrame for light theme
styled_df = df.style.format({
    'cost': '{:.6f}',
    'latency': '{:.6f}',
    'call_counter': '{:d}',
    'input_tokens': '{:d}',
    'output_tokens': '{:d}'
}).set_table_styles([
    {'selector': 'th', 'props': [
        ('background-color', '#6C5CE7'),
        ('color', 'white'),
        ('font-weight', 'bold'),
        ('text-align', 'center'),
        ('border', '2px solid #B2A4FF'),
        ('padding', '10px')
    ]},
    {'selector': 'td', 'props': [
        ('background-color', '#FAFAFA'),
        ('color', '#2B2D42'),
        ('text-align', 'center'),
        ('border', '1px solid #B2A4FF'),
        ('padding', '10px')
    ]},
    {'selector': 'tr:nth-child(even)', 'props': [
        ('background-color', '#F0F0F5')
    ]},
    {'selector': 'tbody tr:hover', 'props': [
        ('background-color', '#E0D7F7'),
        ('color', '#6C5CE7')
    ]}
])

# Display styled data and package info side by side
st.markdown("### Detailed Data View", unsafe_allow_html=True)
left_col, right_col = st.columns([2, 1])

with left_col:
    if len(df) > 10:
        if 'show_all_data' not in st.session_state:
            st.session_state['show_all_data'] = False
        if st.session_state['show_all_data']:
            if st.button('Show less data'):
                st.session_state['show_all_data'] = False
            st.markdown(styled_df.to_html(), unsafe_allow_html=True)
        else:
            if st.button('Show all data'):
                st.session_state['show_all_data'] = True
            st.markdown(df.head(10).style.format({
                'cost': '{:.6f}',
                'latency': '{:.6f}',
                'call_counter': '{:d}',
                'input_tokens': '{:d}',
                'output_tokens': '{:d}'
            }).set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#6C5CE7'),
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('border', '2px solid #B2A4FF'),
                    ('padding', '10px')
                ]},
                {'selector': 'td', 'props': [
                    ('background-color', '#FAFAFA'),
                    ('color', '#2B2D42'),
                    ('text-align', 'center'),
                    ('border', '1px solid #B2A4FF'),
                    ('padding', '10px')
                ]},
                {'selector': 'tr:nth-child(even)', 'props': [
                    ('background-color', '#F0F0F5')
                ]},
                {'selector': 'tbody tr:hover', 'props': [
                    ('background-color', '#E0D7F7'),
                    ('color', '#6C5CE7')
                ]}
            ]).to_html(), unsafe_allow_html=True)
    else:
        st.markdown(styled_df.to_html(), unsafe_allow_html=True)

with right_col:
    st.image("/Users/harshikaagarwal/Desktop/newberry/newberry_metrics/newberry_metrics/Screenshot_2025-05-06_at_4.25.24_PM-removebg-preview (1).png", width=680)
    st.markdown(
        """
        <div style='padding-top: 10px;'>
        <h4 style='color:#6C5CE7;'>Newberry Metrics</h4>
        <p style='color:#2C3E50;'><b>What is it?</b><br>
        A lightweight Python package to track cost, latency, and performance metrics of LLMs on Amazon Bedrock.
        </p>
        <p style='color:#2C3E50;'><b>Why use it?</b><br>
        <ul style='color:#2C3E50; list-style-type: none; padding-left: 0;'>
        <li>🔹 Measure model cost per million tokens</li>
        <li>🔹 Get cost of a specific prompt or session</li>
        <li>🔹 Track latency and concurrency in real time</li>
        <li>🔹 Set budget/latency alerts for production use</li>
        <li>🔹 Export metrics per session, hour, or day</li>
        <li>🔹Support Dashboard for Visualization</li>
        </ul>
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
