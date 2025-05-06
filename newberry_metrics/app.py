
import streamlit as st
import pandas as pd
import plotly.express as px
from pandas.io.formats.style import Styler

# Page configuration
st.set_page_config(
    page_title="Entire session",
    page_icon="✅",
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
    </style>
    """,
    unsafe_allow_html=True,
)

# Load CSV data
df = pd.read_csv("/Users/harshikaagarwal/Desktop/newberry/newberry_metrics/newberry_metrics/DEMO.CSV")

# Page title
st.title("Entire session")

# KPI calculations
avg_cost = df['cost'].mean()
total_cost = df['cost'].sum()
avg_latency = df['latency'].mean()
total_latency = df['latency'].sum()

# KPI display
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(label="Avg Cost", value=f"${avg_cost:.2f}")
kpi2.metric(label="Total Cost", value=f"${total_cost:.2f}")
kpi3.metric(label="Avg Latency", value=f"{avg_latency:.2f} ms")
kpi4.metric(label="Total Latency", value=f"{total_latency:.2f} ms")

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

# Line chart: Cost vs Latency
fig2 = px.line(
    df,
    x='latency',
    y='cost',
    title="Cost vs Latency Over Time",
    template=style['plotly_template'],
)
fig2.update_traces(line=dict(color=style['marker_color']))
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
styled_df = df.style.set_table_styles([
    {'selector': 'th', 'props': [
        ('background-color', '#6C5CE7'),
        ('color', 'white'),
        ('font-weight', 'bold'),
        ('text-align', 'center')
    ]},
    {'selector': 'td', 'props': [
        ('background-color', '#FAFAFA'),
        ('color', '#2B2D42'),
        ('text-align', 'center')
    ]},
    {'selector': 'tr:nth-child(even)', 'props': [
        ('background-color', '#F0F0F5')
    ]}
]).set_properties(**{
    'border': '1px solid #E0E0E0',
    'padding': '8px'
})

# Display styled data
st.markdown("### Detailed Data View", unsafe_allow_html=True)
st.markdown(styled_df.to_html(), unsafe_allow_html=True)
