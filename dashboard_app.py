import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# === Config ===
RESULTS_ROOT = "results"

def load_dips(pair, timeframe):
    filename = f"{pair}_{timeframe}_validated_dips.csv"
    path = os.path.join(RESULTS_ROOT, pair, timeframe, filename)
    if not os.path.exists(path):
        st.error(f"No validated dips file found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=['start_time', 'end_time'])
    return df

def filter_dips(df, recent_n=None, start_time=None, end_time=None, session=None):
    filtered = df.copy()

    if session:
        filtered = filtered[filtered['origin_session'].str.lower() == session.lower()]

    if start_time:
        filtered = filtered[filtered['start_time'] >= pd.to_datetime(start_time)]

    if end_time:
        filtered = filtered[filtered['start_time'] <= pd.to_datetime(end_time)]

    if recent_n:
        filtered = filtered.sort_values('start_time', ascending=False).head(recent_n)

    return filtered

def plot_histogram(data, column, title):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data[column], bins=20, kde=True, color='skyblue', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def main():
    st.title("Dip Analysis Dashboard")

    # User inputs in sidebar
    pair = st.sidebar.text_input("Currency Pair", value="GBPJPY")
    timeframe = st.sidebar.text_input("Timeframe", value="M30")

    recent_n = st.sidebar.number_input("Recent N dips (0 = all)", min_value=0, value=100)
    start_time = st.sidebar.date_input("Start Date", value=None)
    end_time = st.sidebar.date_input("End Date", value=None)
    session = st.sidebar.selectbox("Session", options=["", "Asian", "London", "New York"])

    # Convert date inputs to strings for filtering or None if not set
    start_time_str = start_time.isoformat() if start_time else None
    end_time_str = end_time.isoformat() if end_time else None

    df = load_dips(pair, timeframe)
    if df.empty:
        st.stop()

    filtered = filter_dips(
        df,
        recent_n=recent_n if recent_n > 0 else None,
        start_time=start_time_str,
        end_time=end_time_str,
        session=session if session else None,
    )

    st.write(f"Showing {len(filtered)} dips matching filters")

    if len(filtered) == 0:
        st.warning("No dips match the filter criteria.")
        st.stop()

    # Show histograms
    plot_histogram(filtered, "pip_drop", "Distribution of Pip Drops")
    plot_histogram(filtered, "avg_pip_velocity", "Distribution of Average Pip Velocity")

    # Bonus: Show raw data in an expandable section
    with st.expander("Show filtered dips data"):
        st.dataframe(filtered)

if __name__ == "__main__":
    main()
