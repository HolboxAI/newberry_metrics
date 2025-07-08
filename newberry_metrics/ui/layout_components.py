"""
Layout component functions for Newberry Metrics Dashboard.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from newberry_metrics.ui
ui_dir = Path(__file__).parent
package_dir = ui_dir.parent
sys.path.insert(0, str(package_dir.parent))

from newberry_metrics.ui.styles import apply_table_styles, get_table_styles


def display_header(page_icon_path: Path):
    """Display the dashboard header with logo, title and refresh button."""
    # Use 3 columns for Logo | Title | Button
    logo_col, title_col, button_col = st.columns([0.15, 0.7, 0.15])  # Adjust ratios as needed

    with logo_col:
        if page_icon_path.exists():
            st.image(str(page_icon_path), width=190)  # Adjust width for visual balance

    with title_col:
        # Add vertical space using markdown to push title down slightly
        st.markdown("<h1 style='text-align: center; margin-top: -10px;'>Newberry Metrics Dashboard</h1>", 
                   unsafe_allow_html=True)

    with button_col:
        # Add vertical space to align button better
        st.write("")  # Spacer
        if st.button("🔄", key="refresh_button_title"):  # Refresh icon button
            st.rerun()

    st.markdown("---")  # Visual separator


def display_kpis(kpis: dict):
    """Display KPI metrics in columns."""
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(label="Avg Cost", value=f"${kpis['avg_cost']:.6f}")
    kpi2.metric(label="Total Cost", value=f"${kpis['total_cost']:.6f}")
    kpi3.metric(label="Avg Latency", value=f"{kpis['avg_latency']:.6f} ms")
    kpi4.metric(label="Total Latency", value=f"{kpis['total_latency']:.6f} ms")


def display_data_table(df: pd.DataFrame, page_icon_path: Path):
    """Display the detailed data table with package info."""
    st.markdown("### Detailed Data View", unsafe_allow_html=True)
    left_col, right_col = st.columns([2, 1])

    with left_col:
        _display_data_table_content(df)
        
    with right_col:
        _display_package_info(page_icon_path)


def _display_data_table_content(df: pd.DataFrame):
    """Display the data table content."""
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
    elif not df_display.empty:  # If no call_counter, create S.No. starting from 1
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

    # Apply table styles
    apply_table_styles()
    table_styles = get_table_styles()

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
            st.markdown('<div class="scrollable-table">' + styled_content.to_html() + '</div>', unsafe_allow_html=True)
        else:
            df_head = df_display.head(10)
            styled_content = df_head.style.format(active_formats).set_table_styles(table_styles).hide(axis="index")
            st.markdown('<div class="scrollable-table">' + styled_content.to_html() + '</div>', unsafe_allow_html=True)
            st.markdown(f"... {len(df_display) - 10} more rows hidden ...")
    else:
        styled_content = df_display.style.format(active_formats).set_table_styles(table_styles).hide(axis="index")
        st.markdown('<div class="scrollable-table">' + styled_content.to_html() + '</div>', unsafe_allow_html=True)


def _display_package_info(page_icon_path: Path):
    """Display package information sidebar."""
    if page_icon_path.exists():
        st.image(str(page_icon_path), width=400)  # Adjust width as needed
    st.markdown(
        """
        <div style='padding-top: 20px;'>
        <h4 style='color:#6C5CE7;'>Newberry</h4>
        <p style='color:#2C3E50;'><b>What is it?</b><br>
        newberry-metrics is a lightweight Python package designed to track the cost, latency, and performance metrics of LLMs (Large Language Models) like Nova Micro and Claude 3.5 Sonnet from Amazon Bedrock — all with just one or two lines of code.
        </p>
        <p style='color:#2C3E50;'><b>Why use it?</b><br>
        <ul style='color:#2C3E50; list-style-type: none; padding-left: 0;'>
        <li>🔹 Welcome to Newberry Metrics!</li>
        <li>🔹 Measure model cost per million tokens</li>
        <li>🔹 Get cost of a specific prompt or session</li>
        <li>🔹 Track latency and concurrency in real time</li>
        <li>🔹 Set budget/latency alerts for production use</li>
        <li>🔹 Export metrics per session, hour, or day</li>
        <li>🔹Support Dashboard for Visualization</li>
        </ul>
        </p>
        <p style='color:#2C3E50;'>

        </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_welcome_message():
    """Display welcome message when no data is available."""
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
