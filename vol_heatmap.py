import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import streamlit as st
from typing import Optional, List, Dict
import io
import os


class TrueRangeHeatmapDashboard:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.current_timeframe = None
        self.pip_multipliers = {
            'GBPJPY': 100,  # JPY pairs
            'USDJPY': 100,
            'EURJPY': 100,
            'AUDJPY': 100,
            'CADJPY': 100,
            'CHFJPY': 100,
            'NZDJPY': 100,
            'XAUUSD': 10,  # Gold
            'XAGUSD': 10,  # Silver
            # Default for other pairs (USD, EUR, GBP, etc.)
            'DEFAULT': 10000
        }

    def get_pip_multiplier(self, pair: str) -> int:
        """Get the pip multiplier for a given currency pair"""
        pair = pair.upper()

        # Check if it's a JPY pair
        if 'JPY' in pair:
            return 100

        # Check if it's Gold or Silver
        if pair in self.pip_multipliers:
            return self.pip_multipliers[pair]

        # Default for regular pairs
        return self.pip_multipliers['DEFAULT']

    def load_merged_data(self, file_path_or_buffer, timezone: str = 'UTC', pair: str = None):
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
            self.current_pair = pair
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

        self.current_timeframe = timeframe
        return self.load_merged_data(merged_file, timezone, pair)

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
        """Prepare data in heatmap format with trading day starting at 21:00 UTC"""
        if self.processed_data is None:
            return None

        df = self.processed_data.copy()

        # Create a trading day column (21:00 UTC starts a new trading day)
        # First convert to UTC for consistent trading day calculation
        df['datetime_utc'] = df['datetime'].dt.tz_convert('UTC')

        # 21:00 UTC onwards belongs to the NEXT trading day
        # So 21:00 UTC on July 21st is the START of July 22nd trading day
        df['trading_date'] = df['datetime_utc'].apply(
            lambda x: (x + timedelta(days=1)).date() if x.hour >= 21 else x.date()
        )

        # Calculate trading period based on timeframe
        if self.current_timeframe == 'M30':
            # For M30, create 30-minute periods (0-47 where 0 = 21:00-21:30 UTC)
            df['trading_period'] = ((df['datetime_utc'].dt.hour - 21) % 24) * 2 + (df['datetime_utc'].dt.minute // 30)
        elif self.current_timeframe == 'H4':
            # For H4, create 4-hour periods (0-5 where 0 = 21:00-01:00 UTC)
            df['trading_period'] = ((df['datetime_utc'].dt.hour - 21) % 24) // 4
        else:
            # Default to hourly (H1, D1, etc.)
            df['trading_period'] = (df['datetime_utc'].dt.hour - 21) % 24

        # Filter by date range if provided (filter on trading_date, not original date)
        if start_date:
            df = df[df['trading_date'] >= pd.to_datetime(start_date).date()]
        if end_date:
            # Add one day to end_date to ensure we capture the full Friday
            end_date_adj = pd.to_datetime(end_date).date() + timedelta(days=1)
            df = df[df['trading_date'] <= end_date_adj]

        # Create pivot table using trading dates and periods
        heatmap_data = df.pivot_table(
            values='true_range',
            index='trading_date',
            columns='trading_period',
            aggfunc='mean'
        )

        # Ensure all periods are present based on timeframe
        if self.current_timeframe == 'M30':
            all_periods = list(range(48))  # 48 30-minute periods in 24 hours
        elif self.current_timeframe == 'H4':
            all_periods = list(range(6))  # 6 4-hour periods in 24 hours
        else:
            all_periods = list(range(24))  # 24 hourly periods

        for period in all_periods:
            if period not in heatmap_data.columns:
                heatmap_data[period] = np.nan

        # Sort columns (periods)
        heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)

        # Filter to only include Monday-Friday (0-4 are weekdays)
        heatmap_data = heatmap_data[heatmap_data.index.to_series().apply(lambda x: x.weekday() <= 4)]

        # Sort by date to ensure proper ordering
        heatmap_data = heatmap_data.sort_index()

        return heatmap_data

    def create_heatmap(self, heatmap_data, timezone: str = 'UTC', title: str = "True Range Volatility Heatmap",
                       pair: str = None):
        """Create the heatmap visualization with pip values displayed in cells"""

        # Get pip multiplier for the current pair
        pip_multiplier = self.get_pip_multiplier(pair) if pair else self.get_pip_multiplier('DEFAULT')

        # Convert True Range values to pips
        pip_values = heatmap_data * pip_multiplier

        # Create text matrix for displaying pip values
        text_matrix = []
        for row in pip_values.values:
            text_row = []
            for val in row:
                if pd.isna(val):
                    text_row.append('')
                else:
                    text_row.append(f'{int(round(val))}')
            text_matrix.append(text_row)

        # Create custom colorscale (yellow to red)
        colorscale = [
            [0.0, '#FFFF99'],  # Light yellow
            [0.2, '#FFCC00'],  # Yellow
            [0.4, '#FF9900'],  # Orange
            [0.6, '#FF6600'],  # Dark orange
            [0.8, '#FF3300'],  # Red-orange
            [1.0, '#CC0000']  # Dark red
        ]

        # Create time labels based on timezone and timeframe
        tz = pytz.timezone(timezone)

        # Use a reference date from the heatmap data to get proper DST handling
        reference_date = heatmap_data.index[0] if len(heatmap_data) > 0 else datetime.now().date()

        x_labels = []

        if self.current_timeframe == 'M30':
            # For M30, show all 48 periods
            for period in heatmap_data.columns:
                hour = (period // 2 + 21) % 24
                minute = (period % 2) * 30

                # Create UTC time
                if hour < 21:
                    ref_dt = datetime.combine(reference_date + timedelta(days=1), datetime.min.time())
                else:
                    ref_dt = datetime.combine(reference_date, datetime.min.time())

                utc_dt = ref_dt.replace(hour=hour, minute=minute, tzinfo=pytz.UTC)
                local_dt = utc_dt.astimezone(tz)

                # Show all labels
                x_labels.append(f"{local_dt.hour:02d}:{local_dt.minute:02d}")

        elif self.current_timeframe == 'H4':
            # For H4, show all 6 periods
            for period in heatmap_data.columns:
                start_hour = (period * 4 + 21) % 24
                end_hour = ((period + 1) * 4 + 21) % 24

                # Create UTC time for start of period
                if start_hour < 21:
                    ref_dt = datetime.combine(reference_date + timedelta(days=1), datetime.min.time())
                else:
                    ref_dt = datetime.combine(reference_date, datetime.min.time())

                utc_dt = ref_dt.replace(hour=start_hour, tzinfo=pytz.UTC)
                local_dt = utc_dt.astimezone(tz)

                # Create end time
                if end_hour < 21:
                    ref_dt_end = datetime.combine(reference_date + timedelta(days=1), datetime.min.time())
                else:
                    ref_dt_end = datetime.combine(reference_date, datetime.min.time())

                utc_dt_end = ref_dt_end.replace(hour=end_hour, tzinfo=pytz.UTC)
                local_dt_end = utc_dt_end.astimezone(tz)

                x_labels.append(f"{local_dt.hour:02d}:00-{local_dt_end.hour:02d}:00")

        else:
            # Default hourly labels
            for period in heatmap_data.columns:
                hour = (period + 21) % 24

                if hour < 21:
                    ref_dt = datetime.combine(reference_date + timedelta(days=1), datetime.min.time())
                else:
                    ref_dt = datetime.combine(reference_date, datetime.min.time())

                utc_dt = ref_dt.replace(hour=hour, tzinfo=pytz.UTC)
                local_dt = utc_dt.astimezone(tz)
                x_labels.append(f"{local_dt.hour:02d}:00")

        # Adjust text size based on number of columns
        if self.current_timeframe == 'M30':
            text_size = 8
        elif self.current_timeframe == 'H4':
            text_size = 12
        else:
            text_size = 10

        # Create y-axis labels with day of week and date
        y_labels = []
        y_tickvals = []

        # Group dates by week
        dates_by_week = {}
        for i, date in enumerate(heatmap_data.index):
            week_start = date - timedelta(days=date.weekday())  # Monday of that week
            if week_start not in dates_by_week:
                dates_by_week[week_start] = []
            dates_by_week[week_start].append((i, date))

        # Create labels - the index order matches the heatmap_data order
        for i, date in enumerate(heatmap_data.index):
            day_name = date.strftime('%A')[:3]  # Mon, Tue, etc.
            date_str = date.strftime('%d/%m/%Y')
            y_labels.append(f"{day_name} {date_str}")
            y_tickvals.append(i)

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=list(range(len(x_labels))),
            y=list(range(len(y_labels))),
            text=text_matrix,
            texttemplate='%{text}',
            textfont={"size": text_size, "color": "black"},
            colorscale=colorscale,
            hoverongaps=False,
            hovertemplate='<b>Date:</b> %{customdata}<br>' +
                          '<b>Time:</b> %{x}<br>' +
                          '<b>True Range:</b> %{z:.4f}<br>' +
                          '<b>Pips:</b> %{text}<br>' +
                          '<extra></extra>',
            customdata=[[label] * len(x_labels) for label in y_labels],
            showscale=False  # Remove color scale
        ))

        # Add horizontal lines between weeks
        shapes = []
        # Sort week starts to match our data order
        sorted_week_starts = sorted(dates_by_week.keys(), reverse=True)

        for week_start in sorted_week_starts:
            week_dates = dates_by_week[week_start]
            if week_dates:
                last_day_idx = week_dates[-1][0]
                if last_day_idx < len(heatmap_data) - 1:  # Not the last row
                    shapes.append(
                        dict(
                            type="line",
                            x0=-0.5,
                            x1=len(x_labels) - 0.5,
                            y0=last_day_idx + 0.5,
                            y1=last_day_idx + 0.5,
                            line=dict(color="white", width=2),
                        )
                    )

        # Adjust layout based on timeframe
        if self.current_timeframe == 'M30':
            tick_angle = 90
        else:
            tick_angle = 0

        fig.update_layout(
            title=f"{title} ({timezone})",
            xaxis=dict(
                title=f"Time of Day ({timezone})",
                tickmode='array',
                tickvals=list(range(len(x_labels))),
                ticktext=x_labels,
                side='top',
                tickangle=tick_angle
            ),
            yaxis=dict(
                title="",
                tickmode='array',
                tickvals=y_tickvals,
                ticktext=y_labels
            ),
            height=max(600, len(heatmap_data) * 30),
            font=dict(size=10),
            shapes=shapes,
            plot_bgcolor='rgba(240,240,240,0.5)'
        )

        return fig

    def get_monday_friday_pairs(self, start_date, end_date):
        """Get all Monday-Friday pairs within the date range"""
        pairs = []
        current = start_date

        # Find first Monday
        while current.weekday() != 0:  # 0 is Monday
            current += timedelta(days=1)

        # Collect all Monday-Friday pairs
        while current <= end_date:
            friday = current + timedelta(days=4)
            if friday <= end_date:
                pairs.append((current, friday))
            current += timedelta(days=7)

        return pairs


def main():
    st.set_page_config(page_title="True Range Volatility Heatmap Dashboard", layout="wide")

    st.title("🔥 True Range Volatility Heatmap Dashboard")
    st.markdown("Load your merged trading data to create volatility heatmaps")

    # Initialize session state for data persistence
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = TrueRangeHeatmapDashboard()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'selected_pair' not in st.session_state:
        st.session_state.selected_pair = None
    if 'selected_timeframe' not in st.session_state:
        st.session_state.selected_timeframe = None

    dashboard = st.session_state.dashboard

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

    if data_method == "Load from Data Folder":
        st.sidebar.subheader("Data Folder Settings")

        # Data root path
        data_root = st.sidebar.text_input(
            "Data Root Path",
            value="data",
            help="Path to your data root folder"
        )

        # Pair selection
        pairs = ["GBPUSD", "GBPJPY", "GBPAUD", "XAUUSD"]  # Added Gold
        selected_pair = st.sidebar.selectbox(
            "Select Currency Pair",
            options=pairs
        )

        # Timeframe selection
        timeframes = ["M30", "H1", "H4", "D1"]
        selected_timeframe = st.sidebar.selectbox(
            "Select Timeframe",
            options=timeframes
        )

        # Load button
        if st.sidebar.button("Load Data"):
            st.session_state.data_loaded = dashboard.load_from_data_folder(
                selected_pair,
                selected_timeframe,
                data_root,
                selected_timezone
            )
            if st.session_state.data_loaded:
                st.session_state.selected_pair = selected_pair
                st.session_state.selected_timeframe = selected_timeframe
                st.success(f"✅ Loaded {selected_pair} {selected_timeframe} data successfully!")

    else:  # Upload CSV File
        # Add manual pair input for uploaded files
        selected_pair = st.sidebar.text_input(
            "Currency Pair (for pip calculation)",
            value="GBPUSD",
            help="Enter the currency pair to calculate pips correctly (e.g., GBPUSD, GBPJPY, XAUUSD)"
        )

        # Add timeframe selection for uploaded files
        upload_timeframe = st.sidebar.selectbox(
            "Select Timeframe",
            options=["M30", "H1", "H4", "D1"],
            index=1,  # Default to H1
            help="Select the timeframe of your uploaded data"
        )

        uploaded_file = st.file_uploader(
            "Upload Merged CSV file",
            type=['csv'],
            help="Expected columns: time, open, high, low, close (and optionally volume)"
        )

        if uploaded_file is not None:
            dashboard.current_timeframe = upload_timeframe
            st.session_state.data_loaded = dashboard.load_merged_data(uploaded_file, selected_timezone, selected_pair)
            if st.session_state.data_loaded:
                st.session_state.selected_pair = selected_pair
                st.session_state.selected_timeframe = upload_timeframe
                st.success("✅ Data loaded successfully!")

    # Process data if loaded
    if st.session_state.data_loaded:
        # Calculate True Range
        tr_data = dashboard.calculate_true_range()

        if tr_data is not None:
            # Show data info
            st.info(f"📊 Data loaded: {len(tr_data)} records from {tr_data['date'].min()} to {tr_data['date'].max()}")

            # Date range selection with Monday-Friday constraint
            min_date = tr_data['date'].min()
            max_date = tr_data['date'].max()

            # Get available Monday-Friday pairs
            week_pairs = dashboard.get_monday_friday_pairs(min_date, max_date)

            if week_pairs:
                st.sidebar.subheader("Week Selection")
                st.sidebar.info("Select complete Monday-Friday weeks for proper visualization")

                # Create week options
                week_options = []
                for monday, friday in week_pairs:
                    week_str = f"{monday.strftime('%d/%m/%Y')} - {friday.strftime('%d/%m/%Y')}"
                    week_options.append(week_str)

                # Default to last 4 weeks if available
                default_weeks = min(4, len(week_options))
                selected_weeks = st.sidebar.multiselect(
                    "Select Weeks",
                    options=week_options,
                    default=week_options[-default_weeks:] if week_options else []
                )

                if selected_weeks:
                    # Parse selected weeks to get date range
                    selected_indices = [week_options.index(week) for week in selected_weeks]
                    selected_pairs = [week_pairs[i] for i in selected_indices]

                    # Prepare heatmap data for each selected week
                    all_heatmap_data = []

                    # Sort selected pairs by date (most recent first) before processing
                    selected_pairs_sorted = sorted(selected_pairs, key=lambda x: x[0], reverse=True)

                    for monday, friday in selected_pairs_sorted:
                        week_data = dashboard.prepare_heatmap_data(
                            monday.strftime('%Y-%m-%d'),
                            friday.strftime('%Y-%m-%d')
                        )
                        if week_data is not None and not week_data.empty:
                            all_heatmap_data.append(week_data)

                    if all_heatmap_data:
                        # Combine all selected weeks
                        heatmap_data = pd.concat(all_heatmap_data)

                        # Remove any duplicate dates and sort by date descending (most recent first)
                        heatmap_data = heatmap_data[~heatmap_data.index.duplicated(keep='first')]
                        heatmap_data = heatmap_data.sort_index(ascending=False)

                        # Create and display heatmap
                        chart_title = "True Range Volatility Heatmap"
                        if st.session_state.selected_pair:
                            chart_title = f"True Range Volatility Heatmap - {st.session_state.selected_pair} {st.session_state.selected_timeframe if data_method == 'Load from Data Folder' else ''}"

                        fig = dashboard.create_heatmap(
                            heatmap_data,
                            selected_timezone,
                            chart_title,
                            st.session_state.selected_pair
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Show data preview
                        with st.expander("📋 Data Preview"):
                            st.dataframe(heatmap_data.head(10))

                    else:
                        st.warning("No data available for the selected weeks.")
                else:
                    st.warning("Please select at least one week to visualize.")
            else:
                st.warning("No complete Monday-Friday weeks found in the data.")

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
└── XAUUSD/
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