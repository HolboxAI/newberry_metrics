import streamlit as st
import pandas as pd
import plotly.express as px
from pandas.io.formats.style import Styler
from typing import Optional
import json
from pathlib import Path
import os
import glob

APP_SCRIPT_DIR = Path(__file__).parent.resolve()

# --- Configuration for Dynamic Paths & Session Files ---
# Image path (assuming 'logo.png' is in the same directory as app.py)
image_file_name = "logo.png"
page_icon_path = APP_SCRIPT_DIR / image_file_name

# Session files are expected to be in the same directory as this app.py script
SESSION_FILES_DIRECTORY = APP_SCRIPT_DIR
SESSION_FILE_PATTERN = "session_metrics_*.json"

# Page configuration
st.set_page_config(
    page_title="Entire session",
    page_icon=str(page_icon_path) if page_icon_path.exists() else "📊",
    layout="wide",
)

# Light theme style settings
style = {
    "plotly_template": "plotly_white",
    "line_color": "#6C5CE7",       # Elegant violet
    "marker_color": "#FDCB6E",     # Soft amber
    "bg_color": "#FAFAFA",         # Light, gentle gray-white
    "sidebar_color": "#F0F2F6",    # Muted light blue-gray
    "text_color": "#2C3E50",       # Rich dark blue-gray
    "chart_bgcolor": "#FFFFFF"     # Pure white for high clarity
}

# Apply custom light theme via CSS
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {style['bg_color']};
            color: {style['text_color']};
        }}
        .css-1d391kg, .css-1v3fvcr, .css-hxt7ib {{
            background-color: {style['sidebar_color']} !important;
        }}
        h1, h2, h3, h4, h5, h6, p, div {{
            color: {style['text_color']} !important;
        }}
        .stDataFrame, .stTable {{
            background-color: {style['chart_bgcolor']} !important;
        }}
        div.stButton > button,
        div.stButton > button:active,
        div.stButton > button:focus,
        div.stButton > button:hover {{
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
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Data Loading Functions ---
def find_latest_session_file(directory: Path, pattern: str) -> Optional[Path]:
    """Finds the most recently modified file matching the pattern in the specified directory."""
    try:
        session_files = list(directory.glob(pattern))
        if not session_files:
            return None
        latest_file = max(session_files, key=lambda p: p.stat().st_mtime)
        return latest_file
    except Exception as e:
        st.error(f"Error finding session files in {directory}: {e}")
        return None

def load_session_data(file_path: Path) -> Optional[dict]:
    """Loads session data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {file_path}. File might be corrupted or empty.", icon="⚠️")
        return None
    except IOError:
        st.error(f"Error: Could not read session file: {file_path}", icon="❌")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading {file_path}: {e}", icon="🔥")
        return None

latest_session_file_path = find_latest_session_file(SESSION_FILES_DIRECTORY, SESSION_FILE_PATTERN)
session_data = None
df = pd.DataFrame() # Initialize empty DataFrame

if latest_session_file_path:
    session_data = load_session_data(latest_session_file_path)
    if session_data and "api_calls" in session_data:
        if session_data["api_calls"]: # Check if list is not empty
            df = pd.DataFrame(session_data["api_calls"])
            # Ensure essential columns exist and convert timestamp
            expected_cols = {'cost': 0.0, 'latency': 0.0, 'input_tokens': 0,
                             'output_tokens': 0, 'call_counter': 0, 'timestamp': pd.NaT}
            for col, default_val in expected_cols.items():
                if col not in df.columns:
                    df[col] = default_val
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if 'call_counter' not in df.columns and not df.empty:
                df['call_counter'] = range(1, len(df) + 1)
            # Ensure numeric types
            for col in ['cost', 'latency', 'input_tokens', 'output_tokens', 'call_counter']:
                 if col in df.columns:
                      df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    elif session_data is None: # Error during loading
        pass # Error message handled by load_session_data
    else: # session_data loaded but "api_calls" key missing or None
        st.error("Session file format is unexpected. Missing 'api_calls' data.", icon="❗")
        session_data = None # Ensure consistency
else:
    st.warning(f"No session data files ('{SESSION_FILE_PATTERN}') found in '{SESSION_FILES_DIRECTORY}'.", icon="📁")
    st.markdown("Please run your main application script to generate metrics or refresh if new data is expected.")


# --- Page Title & Logo ---
# Use 3 columns for Logo | Title | Button
logo_col, title_col, button_col = st.columns([0.15, 0.7, 0.15]) # Adjust ratios as needed

with logo_col:
    if page_icon_path.exists():
        st.image(str(page_icon_path), width=190) # Adjust width for visual balance

with title_col:
    # Add vertical space using markdown to push title down slightly
    st.markdown("<h1 style='text-align: center; margin-top: -10px;'>Entire session</h1>", unsafe_allow_html=True)
    # st.title("Entire session") # Using markdown for centering and fine-tuning alignment

with button_col:
    # Add vertical space to align button better
    st.write("") # Spacer
    if st.button("🔄", key="refresh_button_title"): # Refresh icon button
        st.rerun()


# --- KPI calculations --- (This section should use df loaded from JSON)
# Ensure 'df' is populated correctly from session data before this block
if not df.empty:
avg_cost = df['cost'].mean()
total_cost = df['cost'].sum()
    avg_latency = df['latency'].mean() # Assuming latency is in ms if label is ms
total_latency = df['latency'].sum()
else:
    # Set default values if df is empty
    avg_cost = 0.0
    total_cost = 0.0
    avg_latency = 0.0
    total_latency = 0.0

# --- KPI display ---
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
    if not df.empty: # Check if df has data before proceeding
        # --- Prepare DataFrame for display ---
        # Create a copy and ensure 'call_counter' is correctly numbered from 1
        df_display = df.copy()
        if 'call_counter' in df_display.columns:
            df_display['call_counter'] = df_display['call_counter'].fillna(0).astype(int)
            # Ensure call_counter starts from 1 if it was originally 0-based or missing
            if (df_display['call_counter'] == 0).all() or (df_display['call_counter'].iloc[0] != 1 if not df_display.empty else False):
                 if not df_display.empty:
                    df_display['call_counter'] = range(1, len(df_display) + 1)
            # Rename 'call_counter' to 'S.No.' for display
            df_display = df_display.rename(columns={'call_counter': 'S.No.'})
        else:
            # If no call_counter, create S.No. starting from 1
             if not df_display.empty:
                df_display['S.No.'] = range(1, len(df_display) + 1)

        # --- Select and Order Columns for Display ---
        # Define the desired order, including S.No.
        display_columns = ['S.No.', 'timestamp', 'cost', 'latency', 'input_tokens', 'output_tokens']
        # Filter to only include columns that actually exist in df_display
        actual_display_columns = [col for col in display_columns if col in df_display.columns]
        df_display = df_display[actual_display_columns] # Reorder and select columns

        # --- Define Formats ---
        cols_to_format = {
                'cost': '{:.6f}',
            'latency': '{:.6f}', # Keep original formatting if latency is in ms
            'S.No.': '{:d}',     # Format S.No. as integer
                'input_tokens': '{:d}',
            'output_tokens': '{:d}',
            'timestamp': lambda t: pd.to_datetime(t).strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(t) else ''
        }
        # Filter formats for columns present in the final df_display
        active_formats = {col: fmt for col, fmt in cols_to_format.items() if col in df_display.columns}

        # --- Define Table Styles ---
        table_styles = [
            {'selector': 'th', 'props': [ # Style all headers, including S.No.
                ('background-color', '#6C5CE7'), ('color', 'white'), ('font-weight', 'bold'),
                ('text-align', 'center'), ('border', '1px solid #ddd'), ('padding', '8px')
                ]},
                {'selector': 'td', 'props': [
                ('background-color', '#FAFAFA'), ('color', '#2B2D42'), ('text-align', 'center'),
                ('border', '1px solid #ddd'), ('padding', '8px')
            ]},
            {'selector': 'tr:nth-child(even) td', 'props': [
                    ('background-color', '#F0F0F5')
                ]},
            {'selector': 'tbody tr:hover td', 'props': [
                ('background-color', '#E0D7F7 !important'), ('color', '#6C5CE7 !important')
            ]}
            # Removed index-specific styling as we are hiding the default index
        ]

        # --- Logic for displaying all/less data ---
        if len(df_display) > 10:
            if 'show_all_data' not in st.session_state:
                st.session_state['show_all_data'] = False
            
            show_all = st.session_state['show_all_data']
            button_label = 'Show less data' if show_all else 'Show all data'
            
            if st.button(button_label, key="show_hide_detail_button"):
                st.session_state['show_all_data'] = not show_all
                st.rerun()

            if show_all:
                # Display full table, hide the default 0-based index
                styled_content = df_display.style.format(active_formats).set_table_styles(table_styles).hide(axis="index")
                st.markdown(styled_content.to_html(), unsafe_allow_html=True)
            else:
                # Display head only, hide the default 0-based index
                df_head = df_display.head(10)
                styled_content = df_head.style.format(active_formats).set_table_styles(table_styles).hide(axis="index")
                st.markdown(styled_content.to_html(), unsafe_allow_html=True)
                st.markdown(f"... {len(df_display) - 10} more rows hidden ...")

        else:
            # Display full table (less than 10 rows), hide the default 0-based index
            styled_content = df_display.style.format(active_formats).set_table_styles(table_styles).hide(axis="index")
            st.markdown(styled_content.to_html(), unsafe_allow_html=True)
    else:
         st.info("No detailed API call data to display.", icon="💾")

with right_col:
    if page_icon_path.exists():
        st.image(str(page_icon_path), width=400) # Adjust width as needed
    st.markdown(
        """
        <div style='padding-top: 20px;'>
        <h4 style='color:#6C5CE7;'>Newberry</h4>
        <p style='color:#2C3E50;'><b>What is it?</b><br>
        newberry-metrics is a lightweight Python package designed to track the cost, latency, and performance metrics of LLMs (Large Language Models) like Nova Micro and Claude 3.5 Sonnet from Amazon Bedrock — all with just one or two lines of code.
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
        <b>Version:</b> 1.0.5<br>
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )
