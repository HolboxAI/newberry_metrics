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

# NEW: Session files are now expected in the user's home directory
SESSIONS_BASE_DIR = Path.home() / ".newberry_metrics" / "sessions"
SESSION_FILES_DIRECTORY = SESSIONS_BASE_DIR
SESSION_FILE_PATTERN = "session_metrics_*.json"

# Page configuration
st.set_page_config(
    page_title="Newberry Metrics Dashboard",
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

        /* Specific style for the refresh button */
        div[data-testid="stButton-refresh_button_title"] > button,
        div[data-testid="stButton-refresh_button_title"] > button:hover,
        div[data-testid="stButton-refresh_button_title"] > button:active,
        div[data-testid="stButton-refresh_button_title"] > button:focus {{
            background-color: #ADD8E6 !important; /* Light Blue */
            color: #2C3E50 !important;            /* Dark text for contrast */
            border: 1px solid #87CEEB !important; /* Slightly darker/complementary blue border */
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Data Loading Functions ---
def find_latest_session_file(directory: Path, pattern: str) -> Optional[Path]:
    """Finds the most recently modified file matching the pattern in the specified directory."""
    try:
        # Ensure the directory exists before trying to glob from it
        if not directory.exists():
            # Optionally: st.info(f"Metrics directory {directory} not found. Waiting for initial metrics generation.")
            return None
        directory.mkdir(parents=True, exist_ok=True) # Ensure it exists, useful if app starts before any metrics saved
        
        session_files = list(directory.glob(pattern))
        if not session_files:
            return None
        latest_file = max(session_files, key=lambda p: p.stat().st_mtime)
        return latest_file
    except Exception as e:
        # st.error(f"Error accessing session files directory {directory}: {e}") # Can be noisy
        return None

def load_session_data(file_path: Path) -> Optional[dict]:
    """Loads session data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        return None
    except IOError:
        return None
    except Exception as e:
        return None

# --- Page Title & Logo ---
# Use 3 columns for Logo | Title | Button
logo_col, title_col, button_col = st.columns([0.15, 0.7, 0.15]) # Adjust ratios as needed

with logo_col:
    if page_icon_path.exists():
        st.image(str(page_icon_path), width=190) # Adjust width for visual balance

with title_col:
    # Add vertical space using markdown to push title down slightly
    st.markdown("<h1 style='text-align: center; margin-top: -10px;'>Newberry Metrics Dashboard</h1>", unsafe_allow_html=True)

with button_col:
    # Add vertical space to align button better
    st.write("") # Spacer
    if st.button("🔄", key="refresh_button_title"): # Refresh icon button
        st.rerun()

st.markdown("---") # Visual separator

# At the beginning of the main app logic, before calling find_latest_session_file:
# Ensure the target directory for session files exists or can be created.
# This helps if the app is launched before any session file is created by main.py.
try:
    SESSION_FILES_DIRECTORY.mkdir(parents=True, exist_ok=True)
except Exception as e:
    st.error(f"Could not create or access the session data directory: {SESSION_FILES_DIRECTORY}. Please check permissions. Error: {e}")
    # Potentially exit or display a persistent error state if the directory is critical and cannot be accessed/created.

latest_session_file_path = find_latest_session_file(SESSION_FILES_DIRECTORY, SESSION_FILE_PATTERN)
df = pd.DataFrame() # Initialize df as empty
session_data_loaded_successfully = False # Flag to track if session_data was loaded

if latest_session_file_path:
    session_data = load_session_data(latest_session_file_path) # This function handles its own st.error for load failures

    if session_data:
        session_data_loaded_successfully = True # Mark that we at least loaded a file
        if "api_calls" in session_data and session_data["api_calls"]:
            # Attempt to create and process DataFrame
            try:
                temp_df = pd.DataFrame(session_data["api_calls"])
                # Ensure essential columns exist and convert timestamp
                expected_cols = {'cost': 0.0, 'latency': 0.0, 'input_tokens': 0,
                                 'output_tokens': 0, 'call_counter': 0, 'timestamp': pd.NaT}
                for col, default_val in expected_cols.items():
                    if col not in temp_df.columns:
                        temp_df[col] = default_val
                
                if 'timestamp' in temp_df.columns:
                    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], errors='coerce')
                else:
                    st.error("Critical error: 'timestamp' column missing after DataFrame creation from session data.")
                    temp_df = pd.DataFrame() # Reset to empty

                if not temp_df.empty:
                    if 'call_counter' not in temp_df.columns:
                        temp_df['call_counter'] = range(1, len(temp_df) + 1)
                    # Ensure numeric types
                    for col_numeric in ['cost', 'latency', 'input_tokens', 'output_tokens', 'call_counter']:
                        if col_numeric in temp_df.columns:
                            temp_df[col_numeric] = pd.to_numeric(temp_df[col_numeric], errors='coerce').fillna(0)
                
                df = temp_df # Assign to the main df if processing was successful

            except Exception as e:
                st.error(f"Error processing session data from {latest_session_file_path.name}: {e}")
                df = pd.DataFrame() # Ensure df is empty on processing error
        else: # session_data loaded, but "api_calls" key missing or api_calls list is empty
            st.info(
                f"Session file '{latest_session_file_path.name}' was loaded, but it appears to be empty or does not contain any API call metrics yet.",
                icon="ℹ️"
            )
            st.markdown("Please perform some operations with Bedrock to generate metrics. Then click the refresh button (🔄) above.")
    # If session_data is None here, load_session_data has already shown an error. df remains empty.

if not df.empty:
    # --- KPI calculations --- (This section should use df loaded from JSON)
    avg_cost = df['cost'].mean()
    total_cost = df['cost'].sum()
    avg_latency = df['latency'].mean() 
    total_latency = df['latency'].sum()

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
        # Add a check for division by zero if input_tokens can be zero
        hourly_io['io_ratio'] = hourly_io.apply(lambda row: row['output_tokens'] / row['input_tokens'] if row['input_tokens'] else 0, axis=1)


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
        daily_io['io_ratio'] = daily_io.apply(lambda row: row['output_tokens'] / row['input_tokens'] if row['input_tokens'] else 0, axis=1)


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

    # Display styled data and package info side by side
    st.markdown("### Detailed Data View", unsafe_allow_html=True)
    left_col, right_col = st.columns([2, 1])

    with left_col:
        # --- Prepare DataFrame for display ---
        # Create a copy and ensure 'call_counter' is correctly numbered from 1
        df_display = df.copy()
        if 'call_counter' in df_display.columns:
            df_display['call_counter'] = df_display['call_counter'].fillna(0).astype(int)
            # Ensure call_counter starts from 1 if it was originally 0-based or missing
            if not df_display.empty and ((df_display['call_counter'] == 0).all() or df_display['call_counter'].iloc[0] != 1):
                df_display['call_counter'] = range(1, len(df_display) + 1)
            # Rename 'call_counter' to 'S.No.' for display
            df_display = df_display.rename(columns={'call_counter': 'S.No.'})
        elif not df_display.empty : # If no call_counter, create S.No. starting from 1
            df_display['S.No.'] = range(1, len(df_display) + 1)

        # --- Select and Order Columns for Display ---
        display_columns = ['S.No.', 'timestamp', 'cost', 'latency', 'input_tokens', 'output_tokens']
        actual_display_columns = [col for col in display_columns if col in df_display.columns]
        df_display = df_display[actual_display_columns]

        # --- Define Formats ---
        cols_to_format = {
                'cost': '{:.6f}',
            'latency': '{:.6f}',
            'S.No.': '{:d}',
                'input_tokens': '{:d}',
            'output_tokens': '{:d}',
            'timestamp': lambda t: pd.to_datetime(t).strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(t) else ''
        }
        active_formats = {col: fmt for col, fmt in cols_to_format.items() if col in df_display.columns}

        # --- Define Table Styles ---
        table_styles = [
            {'selector': 'th', 'props': [
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
                styled_content = df_display.style.format(active_formats).set_table_styles(table_styles).hide(axis="index")
                st.markdown(styled_content.to_html(), unsafe_allow_html=True)
            else:
                df_head = df_display.head(10)
                styled_content = df_head.style.format(active_formats).set_table_styles(table_styles).hide(axis="index")
                st.markdown(styled_content.to_html(), unsafe_allow_html=True)
                st.markdown(f"... {len(df_display) - 10} more rows hidden ...")
        else:
            styled_content = df_display.style.format(active_formats).set_table_styles(table_styles).hide(axis="index")
            st.markdown(styled_content.to_html(), unsafe_allow_html=True)
            
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

else: # df is empty
    if not latest_session_file_path: # No session file was found at all
        st.header("👋 Welcome to Newberry Metrics!")
        st.info(
            """
            To see your LLM performance dashboard:

            1.  Use the `newberry_metrics` Python library to make some calls to your Amazon Bedrock models.
                This will automatically save usage data locally.
            2.  Once data is saved, click the refresh button (🔄) at the top right of this page.

            Your metrics will then be visualized here!
            """
        )

    elif session_data_loaded_successfully: # A file was found and loaded, but resulted in an empty df (e.g. empty api_calls)
        # The specific info message for this case is already shown above when 'api_calls' is empty.
        # We can add a general fallback here if needed, or just let the previous messages stand.
        st.markdown("---") # Separator
        st.warning("No metrics data is available to display. Please ensure API calls were made and logged, then refresh.", icon="📊")
    # If latest_session_file_path existed but session_data is None (load_session_data failed),
    # load_session_data itself would have printed an st.error. df is empty, so this block is reached.
    # Adding a generic fallback message if it wasn't the "no file found" or "empty api_calls" case.
    elif latest_session_file_path and not session_data_loaded_successfully:
        st.markdown("---")
        st.error("Could not load or process the session data file. Please check messages above for errors, or try generating new metrics.", icon="❌")
