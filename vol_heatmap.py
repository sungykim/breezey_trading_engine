import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import streamlit as st
from typing import Optional, List
import io
import os


class TrueRangeHeatmapDashboard:
    def __init__(self):
        self.data = None
        self.processed_data = None

    def load_merged_data(self, file_path_or_buffer, timezone: str = 'UTC'):
        """
        Load merged data from your data processing script
        Expected columns: time, open, high, low, close, volume
        """
        try:
            if isinstance(file_path_or_buffer, str):
                df = pd.read_csv(file_path_or_buffer)
            else:
                df = pd.read_csv(file_path_or_buffer)

            # Check for required columns
            required_cols = ['time', 'open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return False

            # Convert time column (assuming it's Unix timestamp)
            df['datetime'] = pd.to_datetime(df['time'], unit='s')

            # Set timezone
            if df['datetime'].dt.tz is None:
                df['datetime'] = df['datetime'].dt.tz_localize('UTC')

            # Convert to specified timezone
            target_tz = pytz.timezone(timezone)
            df['datetime'] = df['datetime'].dt.tz_convert(target_tz)

            self.data = df
            return True

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False

    def load_from_data_folder(self, pair: str, timeframe: str, data_root: str = "data", timezone: str = 'UTC'):
        """
        Load merged data directly from your data folder structure
        """
        merged_file = os.path.join(data_root, pair, timeframe, f"{pair}_{timeframe}_merged.csv")

        if not os.path.exists(merged_file):
            st.error(f"Merged file not found: {merged_file}")
            return False

        return self.load_merged_data(merged_file, timezone)

    def calculate_true_range(self):
        """Calculate True Range for each candle"""
        if self.data is None:
            return None

        df = self.data.copy()

        # Calculate True Range components
        df['h_l'] = df['high'] - df['low']
        df['h_pc'] = abs(df['high'] - df['close'].shift(1))
        df['l_pc'] = abs(df['low'] - df['close'].shift(1))

        # True Range is the maximum of the three components
        df['true_range'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)

        # Add time components for heatmap
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.day_name()
        df['date'] = df['datetime'].dt.date

        self.processed_data = df
        return df

    def prepare_heatmap_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Prepare data in heatmap format similar to your spreadsheet"""
        if self.processed_data is None:
            return None

        df = self.processed_data.copy()

        # Filter by date range if provided
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date).date()]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date).date()]

        # Create pivot table: rows = dates, columns = hours
        heatmap_data = df.pivot_table(
            values='true_range',
            index='date',
            columns='hour',
            aggfunc='mean'  # Average if multiple values per hour
        )

        # Ensure all hours 0-23 are present
        all_hours = list(range(24))
        for hour in all_hours:
            if hour not in heatmap_data.columns:
                heatmap_data[hour] = np.nan

        # Sort columns (hours)
        heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)

        return heatmap_data

    def create_heatmap(self, heatmap_data, timezone: str = 'UTC', title: str = "True Range Volatility Heatmap"):
        """Create the heatmap visualization with yellow-to-red gradient"""

        # Create custom colorscale (yellow to red)
        colorscale = [
            [0.0, '#FFFF99'],  # Light yellow
            [0.2, '#FFCC00'],  # Yellow
            [0.4, '#FF9900'],  # Orange
            [0.6, '#FF6600'],  # Dark orange
            [0.8, '#FF3300'],  # Red-orange
            [1.0, '#CC0000']  # Dark red
        ]

        # Convert index to strings for better display
        y_labels = [str(date) for date in heatmap_data.index]

        # Create hour labels with timezone info
        x_labels = [f"{hour:02d}:00" for hour in heatmap_data.columns]

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            hoverongaps=False,
            hovertemplate='<b>Date:</b> %{y}<br>' +
                          '<b>Hour:</b> %{x}<br>' +
                          '<b>True Range:</b> %{z:.4f}<br>' +
                          '<extra></extra>',
            colorbar=dict(
                title="True Range Value"
            )
        ))

        fig.update_layout(
            title=f"{title} ({timezone})",
            xaxis_title=f"Hour of Day ({timezone})",
            yaxis_title="Date",
            height=max(400, len(heatmap_data) * 25),
            font=dict(size=10),
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1,
                side='top'
            ),
            yaxis=dict(
                autorange='reversed'  # Latest dates at top
            )
        )

        return fig

    def get_statistics(self, heatmap_data):
        """Get basic statistics about the True Range data"""
        flat_data = heatmap_data.values.flatten()
        flat_data = flat_data[~np.isnan(flat_data)]  # Remove NaN values

        return {
            'min': np.min(flat_data),
            'max': np.max(flat_data),
            'mean': np.mean(flat_data),
            'median': np.median(flat_data),
            'std': np.std(flat_data)
        }


def main():
    st.set_page_config(page_title="True Range Volatility Heatmap Dashboard", layout="wide")

    st.title("🔥 True Range Volatility Heatmap Dashboard")
    st.markdown("Load your merged trading data to create volatility heatmaps")

    # Initialize dashboard
    dashboard = TrueRangeHeatmapDashboard()

    # Sidebar for controls
    st.sidebar.header("Configuration")

    # Data loading method selection
    data_method = st.sidebar.radio(
        "Data Loading Method",
        ["Load from Data Folder", "Upload CSV File"],
        help="Choose how to load your data"
    )

    # Timezone selection
    common_timezones = [
        'UTC', 'US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific',
        'Europe/London', 'Europe/Paris', 'Europe/Berlin', 'Europe/Zurich',
        'Asia/Tokyo', 'Asia/Hong_Kong', 'Asia/Singapore', 'Australia/Sydney'
    ]

    selected_timezone = st.sidebar.selectbox(
        "Select Timezone",
        options=common_timezones,
        index=5  # Default to London
    )

    data_loaded = False

    if data_method == "Load from Data Folder":
        st.sidebar.subheader("Data Folder Settings")

        # Data root path
        data_root = st.sidebar.text_input(
            "Data Root Path",
            value="data",
            help="Path to your data root folder"
        )

        # Pair selection
        pairs = ["GBPUSD", "GBPJPY", "GBPAUD"]  # Default pairs from your script
        selected_pair = st.sidebar.selectbox(
            "Select Currency Pair",
            options=pairs
        )

        # Timeframe selection
        timeframes = ["M30", "H1", "H4", "D1"]  # From your script
        selected_timeframe = st.sidebar.selectbox(
            "Select Timeframe",
            options=timeframes
        )

        # Load button
        if st.sidebar.button("Load Data"):
            data_loaded = dashboard.load_from_data_folder(
                selected_pair,
                selected_timeframe,
                data_root,
                selected_timezone
            )
            if data_loaded:
                st.success(f"✅ Loaded {selected_pair} {selected_timeframe} data successfully!")

    else:  # Upload CSV File
        uploaded_file = st.file_uploader(
            "Upload Merged CSV file",
            type=['csv'],
            help="Expected columns: time, open, high, low, close (and optionally volume)"
        )

        if uploaded_file is not None:
            data_loaded = dashboard.load_merged_data(uploaded_file, selected_timezone)
            if data_loaded:
                st.success("✅ Data loaded successfully!")

    # Process data if loaded
    if data_loaded:
        # Calculate True Range
        tr_data = dashboard.calculate_true_range()

        if tr_data is not None:
            # Show data info
            st.info(f"📊 Data loaded: {len(tr_data)} records from {tr_data['date'].min()} to {tr_data['date'].max()}")

            # Date range selection
            min_date = tr_data['date'].min()
            max_date = tr_data['date'].max()

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=max_date - timedelta(days=30),  # Default to last 30 days
                    min_value=min_date,
                    max_value=max_date
                )

            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )

            # Prepare heatmap data
            heatmap_data = dashboard.prepare_heatmap_data(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            if heatmap_data is not None and not heatmap_data.empty:
                # Create and display heatmap
                chart_title = "True Range Volatility Heatmap"
                if data_method == "Load from Data Folder":
                    chart_title = f"True Range Volatility Heatmap - {selected_pair} {selected_timeframe}"

                fig = dashboard.create_heatmap(
                    heatmap_data,
                    selected_timezone,
                    chart_title
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display statistics
                stats = dashboard.get_statistics(heatmap_data)

                st.subheader("📊 True Range Statistics")
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("Minimum", f"{stats['min']:.6f}")
                with col2:
                    st.metric("Maximum", f"{stats['max']:.6f}")
                with col3:
                    st.metric("Mean", f"{stats['mean']:.6f}")
                with col4:
                    st.metric("Median", f"{stats['median']:.6f}")
                with col5:
                    st.metric("Std Dev", f"{stats['std']:.6f}")

                # Show data preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(heatmap_data.tail(10))

            else:
                st.warning("No data available for the selected date range.")

    else:
        # Show sample data format and instructions
        if data_method == "Load from Data Folder":
            st.info("👆 Configure your data folder settings and click 'Load Data' to get started")

            st.subheader("📁 Expected Data Folder Structure")
            st.code("""
data/
├── GBPUSD/
│   ├── M30/
│   │   └── GBPUSD_M30_merged.csv
│   ├── H4/
│   │   └── GBPUSD_H4_merged.csv
│   └── D1/
│       └── GBPUSD_D1_merged.csv
├── GBPJPY/
│   └── ...
└── GBPAUD/
    └── ...
            """)

        else:
            st.info("👆 Please upload your merged CSV file to get started")

        st.subheader("📋 Expected CSV Format")
        sample_data = pd.DataFrame({
            'time': [1704067200, 1704070800, 1704074400],  # Unix timestamps
            'open': [1.1000, 1.1005, 1.1010],
            'high': [1.1008, 1.1012, 1.1015],
            'low': [1.0998, 1.1002, 1.1008],
            'close': [1.1005, 1.1010, 1.1012],
            'volume': [1000, 1200, 800]  # Optional
        })
        st.dataframe(sample_data)


if __name__ == "__main__":
    main()