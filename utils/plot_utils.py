import plotly.graph_objects as go
import pandas as pd

def plot_dips(df, dips, title="Price Chart with Dips", show=True, save_path=None):
    """
    Plot candlestick chart with dip ranges highlighted.

    Parameters:
    - df: pandas DataFrame with OHLC data and datetime_utc
    - dips: list of dip dicts with start_index and end_index
    - title: chart title
    - show: whether to display the chart interactively
    - save_path: if provided, save the plot as HTML
    """

    fig = go.Figure()

    # Plot OHLC candles
    fig.add_trace(go.Candlestick(
        x=df['datetime_utc'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))

    # Highlight dip regions
    for dip in dips:
        start_idx, end_idx = dip['start_index'], dip['end_index']
        dip_range = df.iloc[start_idx:end_idx+1]
        fig.add_vrect(
            x0=dip_range['datetime_utc'].iloc[0],
            x1=dip_range['datetime_utc'].iloc[-1],
            fillcolor='red' if dip['pattern'] == 'A' else 'orange',
            opacity=0.2,
            line_width=0,
            annotation_text=f"Dip {dip['id']} ({dip['pattern']})",
            annotation_position="top left",
        )

    fig.update_layout(
        title=title,
        xaxis_title='Time (UTC)',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=600
    )

    if save_path:
        fig.write_html(save_path)
        print(f"✅ Saved plot to: {save_path}")
    if show:
        fig.show()
