"""
Visualization functions for Newberry Metrics Dashboard.
"""

import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from newberry_metrics.ui
ui_dir = Path(__file__).parent
package_dir = ui_dir.parent
sys.path.insert(0, str(package_dir.parent))

from newberry_metrics.ui.styles import THEME_STYLE


def create_hourly_cost_chart(df: pd.DataFrame):
    """Create a line chart for hourly cost visualization."""
    fig = px.line(
        df,
        x='hour',
        y='cost',
        title="<i>Hourly Cost</i>",
        template=THEME_STYLE['plotly_template'],
    )
    fig.update_traces(line=dict(color=THEME_STYLE['line_color']))
    fig.update_layout(
        plot_bgcolor=THEME_STYLE['chart_bgcolor'],
        paper_bgcolor=THEME_STYLE['chart_bgcolor'],
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
    return fig


def create_hourly_latency_scatter(df: pd.DataFrame):
    """Create a scatter plot for hourly latency visualization."""
    fig = px.scatter(
        df,
        x='hour',
        y='latency',
        title="<i>Hourly Latency</i>",
        template=THEME_STYLE['plotly_template'],
    )
    fig.update_traces(marker=dict(color=THEME_STYLE['marker_color']))
    fig.update_layout(
        plot_bgcolor=THEME_STYLE['chart_bgcolor'],
        paper_bgcolor=THEME_STYLE['chart_bgcolor'],
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
    return fig


def create_hourly_io_bar_chart(df: pd.DataFrame):
    """Create a bar chart for hourly input-output token distribution."""
    fig = px.bar(
        df,
        x='hour',
        y=['input_tokens', 'output_tokens'],
        title="<i>Hourly Input-Output Token Distribution</i>",
        template=THEME_STYLE['plotly_template'],
        barmode='group',
        color_discrete_sequence=['#87CEEB', '#FFE5B4']  # Light blue and light yellow
    )
    fig.update_layout(
        plot_bgcolor=THEME_STYLE['chart_bgcolor'],
        paper_bgcolor=THEME_STYLE['chart_bgcolor'],
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
    return fig


def create_daily_cost_chart(df: pd.DataFrame):
    """Create a line chart for daily cost visualization."""
    fig = px.line(
        df,
        x='day',
        y='cost',
        title="<i>Daily Cost</i>",
        template=THEME_STYLE['plotly_template'],
    )
    fig.update_traces(line=dict(color=THEME_STYLE['line_color']))
    fig.update_layout(
        plot_bgcolor=THEME_STYLE['chart_bgcolor'],
        paper_bgcolor=THEME_STYLE['chart_bgcolor'],
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
    return fig


def create_daily_latency_scatter(df: pd.DataFrame):
    """Create a scatter plot for daily latency visualization."""
    fig = px.scatter(
        df,
        x='day',
        y='latency',
        title="<i>Daily Latency</i>",
        template=THEME_STYLE['plotly_template'],
    )
    fig.update_traces(marker=dict(color=THEME_STYLE['marker_color']))
    fig.update_layout(
        plot_bgcolor=THEME_STYLE['chart_bgcolor'],
        paper_bgcolor=THEME_STYLE['chart_bgcolor'],
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
    return fig


def create_daily_io_bar_chart(df: pd.DataFrame):
    """Create a bar chart for daily input-output token distribution."""
    fig = px.bar(
        df,
        x='day',
        y=['input_tokens', 'output_tokens'],
        title="<i>Daily Input-Output Token Distribution</i>",
        template=THEME_STYLE['plotly_template'],
        barmode='group',
        color_discrete_sequence=['#87CEEB', '#FFE5B4']  # Light blue and light yellow
    )
    fig.update_layout(
        plot_bgcolor=THEME_STYLE['chart_bgcolor'],
        paper_bgcolor=THEME_STYLE['chart_bgcolor'],
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
    return fig
