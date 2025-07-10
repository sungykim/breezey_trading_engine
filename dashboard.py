import streamlit as st
import pandas as pd
import os
import plotly.express as px

RESULTS_ROOT = "results"
PAIRS = ["GBPUSD", "GBPJPY", "GBPAUD"]
TIMEFRAMES = ["D1", "H4", "M30"]

SESSIONS = ['Asian', 'London', 'New York']

def load_data(timeframe):
    dfs = []
    for pair in PAIRS:
        path = os.path.join(RESULTS_ROOT, pair, timeframe, f"{pair}_{timeframe}_validated_dips.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['pair'] = pair
            dfs.append(df)
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined['start_time'] = pd.to_datetime(combined['start_time'])
        combined['year'] = combined['start_time'].dt.year
        combined['month'] = combined['start_time'].dt.month
        combined['month_name'] = combined['start_time'].dt.strftime('%B')
        return combined
    else:
        return pd.DataFrame()

def filter_data(df, pair, month, year, session, time_from, time_to, most_recent_n):
    if not df.empty:
        if pair != "All":
            df = df[df['pair'] == pair]
        if month != "All":
            df = df[df['month_name'] == month]
        if year != "All":
            df = df[df['year'] == int(year)]
        if session != "All":
            df = df[df['origin_session'] == session]
        df = df[(df['start_time'].dt.hour >= time_from) & (df['start_time'].dt.hour <= time_to)]
        if most_recent_n > 0:
            df = df.sort_values('start_time', ascending=False).head(most_recent_n)
    return df

def dip_distribution_table(df):
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby('duration')['pip_drop'].describe()[['count', 'mean', 'std', 'min', 'max']]
    grouped = grouped.rename(columns={
        'count': 'Count',
        'mean': 'Avg Pip Drop',
        'std': 'Std Dev',
        'min': 'Min',
        'max': 'Max'
    })
    return grouped

def main():
    st.title("Breezey Trading Engine Dip Dashboard")

    timeframe = st.selectbox("Select Timeframe", TIMEFRAMES)
    df = load_data(timeframe)

    if df.empty:
        st.warning("No data found for this timeframe.")
        return

    pair_options = ["All"] + PAIRS
    month_options = ["All"] + sorted(df['month_name'].unique().tolist())
    year_options = ["All"] + sorted(df['year'].unique().astype(str).tolist())
    session_options = ["All"] + SESSIONS

    col0, col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1,1])
    with col0:
        pair = st.selectbox("Pair", pair_options, key="pair")
    with col1:
        month = st.selectbox("Month", month_options, key="month")
    with col2:
        year = st.selectbox("Year", year_options, key="year")
    with col3:
        time_range = st.slider("Hour Range", 0, 23, (0, 23), key="time_range")
        time_from, time_to = time_range
    with col4:
        session = st.selectbox("Session", session_options, key="session")
    with col5:
        most_recent_n = st.number_input("Most Recent N", min_value=0, value=0, step=1, key="most_recent")

    filtered_df = filter_data(df, pair, month, year, session, time_from, time_to, most_recent_n)

    st.write(f"Showing {len(filtered_df)} dips after filtering.")

    dist_table = dip_distribution_table(filtered_df)
    st.dataframe(dist_table)

    if not filtered_df.empty:
        with st.expander("View Filtered Raw Data"):
            st.dataframe(filtered_df)

        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv_data,
            file_name=f"filtered_dips_{timeframe}.csv",
            mime='text/csv',
        )

        fig1 = px.histogram(
            filtered_df,
            x='duration',
            nbins=int(filtered_df['duration'].max()),
            title="Distribution of Dips by Bearish Candle Count",
            labels={'duration': 'Number of Bearish Candles', 'count': 'Number of Dips'}
        )
        st.plotly_chart(fig1, use_container_width=True)

        avg_pip_by_duration = filtered_df.groupby('duration')['pip_drop'].mean().reset_index()
        fig2 = px.bar(
            avg_pip_by_duration,
            x='duration',
            y='pip_drop',
            labels={'duration': 'Number of Bearish Candles', 'pip_drop': 'Average Pip Drop'},
            title='Average Pip Drop per Number of Bearish Candles'
        )
        st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
