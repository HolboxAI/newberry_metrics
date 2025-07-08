"""
Styling and theme configuration for the Newberry Metrics Dashboard.
"""

import streamlit as st
from pathlib import Path

# Light theme style settings
THEME_STYLE = {
    "plotly_template": "plotly_white",
    "line_color": "#6C5CE7",       # Elegant violet
    "marker_color": "#FDCB6E",     # Soft amber
    "bg_color": "#FAFAFA",         # Light, gentle gray-white
    "sidebar_color": "#F0F2F6",    # Muted light blue-gray
    "text_color": "#2C3E50",       # Rich dark blue-gray
    "chart_bgcolor": "#FFFFFF"     # Pure white for high clarity
}

def apply_custom_theme():
    """Apply custom light theme via CSS."""
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {THEME_STYLE['bg_color']};
                color: {THEME_STYLE['text_color']};
            }}
            .css-1d391kg, .css-1v3fvcr, .css-hxt7ib {{
                background-color: {THEME_STYLE['sidebar_color']} !important;
            }}
            h1, h2, h3, h4, h5, h6, p, div {{
                color: {THEME_STYLE['text_color']} !important;
            }}
            .stDataFrame, .stTable {{
                background-color: {THEME_STYLE['chart_bgcolor']} !important;
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

def apply_selectbox_theme():
    """Apply theme styling for selectbox components."""
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

def apply_table_styles():
    """Apply CSS for scrollable table container."""
    st.markdown("""
        <style>
            .scrollable-table {
                max-height: 500px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 8px;
                margin: 10px 0;
            }
            .scrollable-table table {
                width: 100%;
                border-collapse: collapse;
            }
            .scrollable-table thead th {
                position: sticky;
                top: 0;
                z-index: 1;
            }
        </style>
        """, unsafe_allow_html=True)

def get_table_styles():
    """Get table styling configuration for pandas DataFrame styling."""
    return [
        {'selector': 'th', 'props': [
            ('background-color', '#6C5CE7'), ('color', 'white'), ('font-weight', 'bold'),
            ('text-align', 'center'), ('border', '1px solid #ddd'), ('padding', '8px'),
            ('position', 'sticky'), ('top', '0'), ('z-index', '1')
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

def configure_page(page_icon_path: Path):
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Newberry Metrics Dashboard",
        page_icon=str(page_icon_path) if page_icon_path.exists() else "📊",
        layout="wide",
    )
