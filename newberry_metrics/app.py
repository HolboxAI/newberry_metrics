import streamlit as st
import pandas as pd
import plotly.express as px
from pandas.io.formats.style import Styler

# Page configuration
st.set_page_config(
    page_title="Entire session",
    page_icon="/Users/harshikaagarwal/Desktop/newberry/newberry_metrics/newberry_metrics/Screenshot_2025-05-06_at_4.25.24_PM-removebg-preview (1).png",
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

# Load CSV data
df = pd.read_csv("/Users/harshikaagarwal/Desktop/newberry/newberry_metrics/newberry_metrics/DEMO.CSV")

# Page title with logo
header_col1, header_col2 = st.columns([1, 10])
with header_col1:
    st.image("/Users/harshikaagarwal/Desktop/newberry/newberry_metrics/newberry_metrics/Screenshot_2025-05-06_at_4.25.24_PM-removebg-preview (1).png", width=200)
with header_col2:
    st.title("Entire session")

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


# Line chart: Total Cost Over Session
fig1 = px.line(
    df,
    y='cost',
    title="Total Cost Over Session",
    template=style['plotly_template'],
)
fig1.update_traces(line=dict(color=style['line_color']))
fig1.update_layout(
    plot_bgcolor=style['chart_bgcolor'],
    paper_bgcolor=style['chart_bgcolor'],
    font=dict(color='#87CEEB'),
    title_font=dict(size=20, color='#87CEEB'),
    xaxis=dict(
        title='Index',
        title_font=dict(color='#87CEEB'),
        tickfont=dict(color='#87CEEB')
    ),
    yaxis=dict(
        title='Cost',
        title_font=dict(color='#87CEEB'),
        tickfont=dict(color='#87CEEB')
    )
)
st.plotly_chart(fig1, use_container_width=True)

# Scatter chart: Cost vs Latency
fig2 = px.scatter(
    df,
    x='latency',
    y='cost',
    title="Cost vs Latency Over Time",
    template=style['plotly_template'],
)
fig2.update_traces(marker=dict(color=style['marker_color']))
fig2.update_layout(
    plot_bgcolor=style['chart_bgcolor'],
    paper_bgcolor=style['chart_bgcolor'],
    font=dict(color='#87CEEB'),
    title_font=dict(size=20, color='#87CEEB'),
    xaxis=dict(
        title='Latency (ms)',
        title_font=dict(color='#87CEEB'),
        tickfont=dict(color='#87CEEB')
    ),
    yaxis=dict(
        title='Cost',
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
        <div style='padding-top: 20px;'>
        <h4 style='color:#6C5CE7;'>Newberry</h4>
        <p style='color:#2C3E50;'><b>What is it?</b><br>
        newberry-metrics is a lightweight Python package designed to track the cost, latency, and performance metrics of LLMs (Large Language Models) like Nova Micro and Claude 3.5 Sonnet from Amazon Bedrock — all with just one or two lines of code.
        </p>
        <p style='color:#2C3E50;'><b>Why use it?</b><br>
        <ul style='color:#2C3E50;'>
        <li>🔹 Measure model cost per million tokens</li>
        <li>🔹 Get cost of a specific prompt or session</li>
        <li>🔹 Track latency and concurrency in real time</li>
        <li>🔹 Set budget/latency alerts for production use</li>
        <li>🔹 Export metrics per session, hour, or day</li>
        <li>🔹 Future support for charts, dashboards, and UI (Gradio/Streamlit)</li>
        </ul>
        </p>
        <p style='color:#2C3E50;'>
        <b>Version:</b> 1.0.5<br>
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )
