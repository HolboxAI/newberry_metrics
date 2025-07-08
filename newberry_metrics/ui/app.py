import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from newberry_metrics.ui
ui_dir = Path(__file__).parent
package_dir = ui_dir.parent
sys.path.insert(0, str(package_dir.parent))

from newberry_metrics.ui.data_loader import find_latest_session_file, load_session_data
from newberry_metrics.ui.styles import apply_custom_theme, apply_selectbox_theme, configure_page
from newberry_metrics.ui.visualizations import (
    create_hourly_cost_chart,
    create_hourly_latency_scatter,
    create_hourly_io_bar_chart,
    create_daily_cost_chart,
    create_daily_latency_scatter,
    create_daily_io_bar_chart
)
from newberry_metrics.ui.kpi_calculator import calculate_kpis, prepare_hourly_data, prepare_daily_data
from newberry_metrics.ui.layout_components import display_header, display_kpis, display_data_table, display_welcome_message
from newberry_metrics.path_utils import get_sessions_directory_with_fallback


APP_SCRIPT_DIR = Path(__file__).parent.resolve()

# --- Configuration for Dynamic Paths & Session Files ---
# Image path (assuming 'logo.png' is in the same directory as app.py)
image_file_name = "logo.png"
page_icon_path = APP_SCRIPT_DIR / image_file_name

# Session files with smart fallback (checks home directory if no files in CWD)
SESSION_FILES_DIRECTORY = get_sessions_directory_with_fallback()
SESSION_FILE_PATTERN = "session_metrics_*.json"


# Main application logic
def main():
    # Page configuration
    configure_page(page_icon_path)

    apply_custom_theme()

    # Display header
    display_header(page_icon_path)

    # At the beginning of the main app logic, before calling find_latest_session_file:
    # Ensure the target directory for session files exists or can be created.
    # This helps if the app is launched before any session file is created by main.py.
    try:
        SESSION_FILES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"Could not create or access the session data directory: {SESSION_FILES_DIRECTORY}. Please check permissions. Error: {e}")
        # Potentially exit or display a persistent error state if the directory is critical and cannot be accessed/created.

    latest_session_file_path = find_latest_session_file(SESSION_FILES_DIRECTORY, SESSION_FILE_PATTERN)
    df = pd.DataFrame()  # Initialize df as empty
    session_data_loaded_successfully = False  # Flag to track if session_data was loaded

    if latest_session_file_path:
        session_data = load_session_data(latest_session_file_path)  # This function handles its own st.error for load failures

        if session_data:
            session_data_loaded_successfully = True  # Mark that we at least loaded a file
            if "api_calls" in session_data and session_data["api_calls"]:
                # Attempt to create and process DataFrame
                try:
                    temp_df = pd.DataFrame(session_data["api_calls"])
                    # Ensure essential columns exist and convert timestamp
                    expected_cols = {
                        'cost': 0.0, 'latency': 0.0, 'input_tokens': 0,
                        'output_tokens': 0, 'call_counter': 0, 'timestamp': pd.NaT
                    }
                    for col, default_val in expected_cols.items():
                        if col not in temp_df.columns:
                            temp_df[col] = default_val

                    if 'timestamp' in temp_df.columns:
                        temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], errors='coerce')
                    else:
                        st.error("Critical error: 'timestamp' column missing after DataFrame creation from session data.")
                        temp_df = pd.DataFrame()  # Reset to empty

                    if not temp_df.empty:
                        if 'call_counter' not in temp_df.columns:
                            temp_df['call_counter'] = range(1, len(temp_df) + 1)
                        # Ensure numeric types
                        for col_numeric in ['cost', 'latency', 'input_tokens', 'output_tokens', 'call_counter']:
                            if col_numeric in temp_df.columns:
                                temp_df[col_numeric] = pd.to_numeric(temp_df[col_numeric], errors='coerce').fillna(0)

                        df = temp_df  # Assign to the main df if processing was successful

                except Exception as e:
                    st.error(f"Error processing session data from {latest_session_file_path.name}: {e}")
                    df = pd.DataFrame()  # Ensure df is empty on processing error
            else:  # session_data loaded, but "api_calls" key missing or api_calls list is empty
                st.info(
                    f"Session file '{latest_session_file_path.name}' was loaded, but it appears to be empty or does not contain any API call metrics yet.",
                    icon="ℹ️"
                )
                st.markdown("Please perform some operations with Bedrock to generate metrics. Then click the refresh button (🔄) above.")
        # If session_data is None here, load_session_data has already shown an error. df remains empty.

    if not df.empty:
        # --- KPI calculations ---
        kpis = calculate_kpis(df)

        # --- KPI display ---
        display_kpis(kpis)

        # Add dropdown menu for view selection
        view_options = ["Hourly View", "Daily View"]
        selected_view = st.selectbox(
            "Select View",
            options=view_options,
            key="view_selector"
        )

        # Style the selectbox to match the theme
        apply_selectbox_theme()

        # --- CHARTS BASED ON DROPDOWN SELECTION ---
        if selected_view == "Hourly View":
            hourly_data = prepare_hourly_data(df)

            # Create and display charts
            st.plotly_chart(create_hourly_io_bar_chart(hourly_data['io_tokens']), use_container_width=True)
            st.plotly_chart(create_hourly_cost_chart(hourly_data['cost']), use_container_width=True)
            st.plotly_chart(create_hourly_latency_scatter(hourly_data['latency']), use_container_width=True)

        elif selected_view == "Daily View":
            daily_data = prepare_daily_data(df)

            # Create and display charts
            st.plotly_chart(create_daily_io_bar_chart(daily_data['io_tokens']), use_container_width=True)
            st.plotly_chart(create_daily_cost_chart(daily_data['cost']), use_container_width=True)
            st.plotly_chart(create_daily_latency_scatter(daily_data['latency']), use_container_width=True)

        # Display styled data and package info side by side
        display_data_table(df, page_icon_path)

    else:  # df is empty
        if not latest_session_file_path:  # No session file was found at all
            display_welcome_message()

        elif session_data_loaded_successfully:  # A file was found and loaded, but resulted in an empty df (e.g. empty api_calls)
            # The specific info message for this case is already shown above when 'api_calls' is empty.
            # We can add a general fallback here if needed, or just let the previous messages stand.
            st.markdown("---")  # Separator
            st.warning("No metrics data is available to display. Please ensure API calls were made and logged, then refresh.", icon="📊")
        # If latest_session_file_path existed but session_data is None (load_session_data failed),
        # load_session_data itself would have printed an st.error. df is empty, so this block is reached.
        # Adding a generic fallback message if it wasn't the "no file found" or "empty api_calls" case.
        elif latest_session_file_path and not session_data_loaded_successfully:
            st.markdown("---")
            st.error("Could not load or process the session data file. Please check messages above for errors, or try generating new metrics.", icon="❌")


if __name__ == "__main__":
    main()

