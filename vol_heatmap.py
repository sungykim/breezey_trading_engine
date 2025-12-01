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

    def prepare_volatility_analysis_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Prepare data for volatility analysis with pip bins"""
        if self.processed_data is None:
            return None

        df = self.processed_data.copy()

        # Convert to UTC for consistent analysis
        df['datetime_utc'] = df['datetime'].dt.tz_convert('UTC')
        df['hour_utc'] = df['datetime_utc'].dt.hour

        # Filter by date range if provided
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date).date()]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date).date()]

        # Convert true range to pips
        pip_multiplier = self.get_pip_multiplier(self.current_pair) if self.current_pair else self.get_pip_multiplier(
            'DEFAULT')
        df['true_range_pips'] = df['true_range'] * pip_multiplier

        return df

    def analyze_hourly_volatility(self, df):
        """Analyze volatility for each hour (0-23 UTC) and create 10-pip bins"""
        results = {}

        for hour in range(24):
            # Filter data for the specific hour
            hour_data = df[df['hour_utc'] == hour]['true_range_pips'].dropna()

            if len(hour_data) == 0:
                continue

            # Create 10-pip bins
            max_pips = hour_data.max()
            num_bins = int(np.ceil(max_pips / 10))

            bins = []
            for i in range(num_bins):
                bin_min = i * 10
                bin_max = (i + 1) * 10 - 1
                bin_data = hour_data[(hour_data >= bin_min) & (hour_data <= bin_max)]

                bins.append({
                    'bin_range': f"{bin_min}-{bin_max}",
                    'bin_min': bin_min,
                    'bin_max': bin_max,
                    'count': len(bin_data),
                    'probability': (len(bin_data) / len(hour_data)) * 100
                })

            # Filter out empty bins
            bins = [b for b in bins if b['count'] > 0]

            hour_name = f"{hour:02d}:00 UTC"
            results[hour_name] = {
                'hour': hour,
                'bins': bins,
                'total_samples': len(hour_data),
                'mean': hour_data.mean(),
                'std': hour_data.std(),
                'min': hour_data.min(),
                'max': hour_data.max(),
                'percentiles': {
                    '25': hour_data.quantile(0.25),
                    '50': hour_data.quantile(0.50),
                    '75': hour_data.quantile(0.75),
                    '95': hour_data.quantile(0.95)
                }
            }

        return results

    def create_hourly_volatility_overview_chart(self, analysis_results):
        """Create overview chart showing key metrics for each hour"""
        hours = []
        means = []
        medians = []
        p95s = []
        samples = []

        for hour_name, data in sorted(analysis_results.items(), key=lambda x: x[1]['hour']):
            hours.append(hour_name)
            means.append(data['mean'])
            medians.append(data['percentiles']['50'])
            p95s.append(data['percentiles']['95'])
            samples.append(data['total_samples'])

        fig = go.Figure()

        # Add mean line
        fig.add_trace(go.Scatter(
            x=hours,
            y=means,
            mode='lines+markers',
            name='Mean',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))

        # Add median line
        fig.add_trace(go.Scatter(
            x=hours,
            y=medians,
            mode='lines+markers',
            name='Median (50th percentile)',
            line=dict(color='green', width=2),
            marker=dict(size=6)
        ))

        # Add 95th percentile line
        fig.add_trace(go.Scatter(
            x=hours,
            y=p95s,
            mode='lines+markers',
            name='95th percentile',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title="Hourly Volatility Overview (UTC)",
            xaxis_title="Hour (UTC)",
            yaxis_title="True Range (Pips)",
            height=500,
            hovermode='x unified',
            xaxis=dict(tickangle=45)
        )

        return fig

    def create_volatility_analysis_charts(self, analysis_results):
        """Create charts for volatility analysis"""
        charts = {}

        for hour_name, data in analysis_results.items():
            if not data['bins']:
                continue

            # Create histogram
            bin_labels = [bin_data['bin_range'] for bin_data in data['bins']]
            probabilities = [bin_data['probability'] for bin_data in data['bins']]

            fig = go.Figure()

            # Add bar chart
            fig.add_trace(go.Bar(
                x=bin_labels,
                y=probabilities,
                name='Probability (%)',
                marker=dict(
                    color=probabilities,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Probability (%)")
                ),
                text=[f"{p:.1f}%" for p in probabilities],
                textposition='auto'
            ))

            fig.update_layout(
                title=f"Volatility Distribution - {hour_name}",
                xaxis_title="Pip Range",
                yaxis_title="Probability (%)",
                showlegend=False,
                height=400
            )

            charts[hour_name] = fig

        return charts

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
    st.markdown("Load your merged trading data to create volatility heatmaps and analyze hourly pip distributions")

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

    # Analysis Mode Selection
    st.sidebar.header("Analysis Mode")
    analysis_mode = st.sidebar.radio(
        "Select Analysis Type",
        ["Heatmap Visualization", "Hourly Volatility Analysis"],
        help="Choose between heatmap view or hourly statistical pip distribution analysis"
    )

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
        pairs = ["AUDCAD", "AUDJPY", "AUDUSD", "CADJPY", "CHFJPY", "EURAUD", "EURCAD", "EURJPY", "EURNZD", "EURUSD",
                 "GBPAUD", "GBPCAD", "GBPJPY", "GBPNZD", "GBPUSD", "NZDCAD", "NZDJPY", "NZDUSD", "USDCAD", "USDJPY",
                 "XAUUSD", "USDCHF"]  # Added Gold
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

            if analysis_mode == "Heatmap Visualization":
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

            else:  # Hourly Volatility Analysis
                st.sidebar.subheader("Analysis Parameters")

                # Date range selection for analysis
                start_date = st.sidebar.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date
                )

                end_date = st.sidebar.date_input(
                    "End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )

                # Analysis options
                st.sidebar.subheader("Analysis Options")

                show_overview = st.sidebar.checkbox("Show Hourly Overview Chart", value=True)
                hours_to_analyze = st.sidebar.multiselect(
                    "Select specific hours for detailed analysis (optional)",
                    options=[f"{h:02d}:00 UTC" for h in range(24)],
                    default=[],
                    help="Leave empty to analyze all hours, or select specific hours for detailed charts"
                )

                if st.sidebar.button("Run Hourly Analysis"):
                    # Prepare data for volatility analysis
                    analysis_data = dashboard.prepare_volatility_analysis_data(
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )

                    if analysis_data is not None and not analysis_data.empty:
                        # Analyze volatility for all hours
                        analysis_results = dashboard.analyze_hourly_volatility(analysis_data)

                        if analysis_results:
                            st.header("⏰ Hourly Volatility Analysis")
                            st.markdown(f"**Analysis Period:** {start_date} to {end_date}")
                            st.markdown(f"**Currency Pair:** {st.session_state.selected_pair}")

                            # Show overview chart if requested
                            if show_overview:
                                st.subheader("📈 Hourly Volatility Overview")
                                overview_chart = dashboard.create_hourly_volatility_overview_chart(analysis_results)
                                st.plotly_chart(overview_chart, use_container_width=True)

                                # Add insights about peak hours
                                mean_by_hour = {data['hour']: data['mean'] for data in analysis_results.values()}
                                peak_hour = max(mean_by_hour, key=mean_by_hour.get)
                                quiet_hour = min(mean_by_hour, key=mean_by_hour.get)

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Most Volatile Hour", f"{peak_hour:02d}:00 UTC",
                                              f"{mean_by_hour[peak_hour]:.1f} pips")
                                with col2:
                                    st.metric("Quietest Hour", f"{quiet_hour:02d}:00 UTC",
                                              f"{mean_by_hour[quiet_hour]:.1f} pips")
                                with col3:
                                    volatility_range = mean_by_hour[peak_hour] - mean_by_hour[quiet_hour]
                                    st.metric("Volatility Range", f"{volatility_range:.1f} pips", "Peak - Quiet")

                            # Summary table for all hours
                            st.subheader("📊 Hourly Statistics Summary")

                            summary_data = []
                            for hour_name, result in sorted(analysis_results.items(), key=lambda x: x[1]['hour']):
                                max_prob_bin = max(result['bins'], key=lambda x: x['probability']) if result[
                                    'bins'] else None
                                low_vol_prob = sum(b['probability'] for b in result['bins'] if b['bin_max'] < 20)
                                high_vol_prob = sum(b['probability'] for b in result['bins'] if b['bin_min'] >= 50)

                                summary_data.append({
                                    'Hour (UTC)': hour_name,
                                    'Samples': result['total_samples'],
                                    'Mean (pips)': round(result['mean'], 2),
                                    'Median (pips)': round(result['percentiles']['50'], 2),
                                    'Max (pips)': round(result['max'], 2),
                                    'Most Common Range': max_prob_bin['bin_range'] if max_prob_bin else 'N/A',
                                    'Low Vol % (<20 pips)': round(low_vol_prob, 1),
                                    'High Vol % (≥50 pips)': round(high_vol_prob, 1),
                                    'Risk Level': 'Low' if result['mean'] < 20 else 'Medium' if result[
                                                                                                    'mean'] < 40 else 'High'
                                })

                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)

                            # Detailed analysis for specific hours
                            if hours_to_analyze:
                                st.subheader("🔍 Detailed Hour Analysis")

                                # Filter results to selected hours
                                filtered_results = {hour: data for hour, data in analysis_results.items()
                                                    if hour in hours_to_analyze}

                                # Create tabs for selected hours
                                if filtered_results:
                                    tabs = st.tabs(list(filtered_results.keys()))

                                    for i, (hour_name, result) in enumerate(filtered_results.items()):
                                        with tabs[i]:
                                            col1, col2 = st.columns([2, 1])

                                            with col1:
                                                # Create histogram chart
                                                charts = dashboard.create_volatility_analysis_charts(
                                                    {hour_name: result})
                                                if hour_name in charts:
                                                    st.plotly_chart(charts[hour_name], use_container_width=True)

                                            with col2:
                                                # Display statistics
                                                st.subheader("Statistics")
                                                st.metric("Total Samples", result['total_samples'])
                                                st.metric("Mean (pips)", f"{result['mean']:.2f}")
                                                st.metric("Std Dev (pips)", f"{result['std']:.2f}")
                                                st.metric("Min (pips)", f"{result['min']:.2f}")
                                                st.metric("Max (pips)", f"{result['max']:.2f}")

                                                st.subheader("Percentiles")
                                                st.metric("25th", f"{result['percentiles']['25']:.2f}")
                                                st.metric("50th (Median)", f"{result['percentiles']['50']:.2f}")
                                                st.metric("75th", f"{result['percentiles']['75']:.2f}")
                                                st.metric("95th", f"{result['percentiles']['95']:.2f}")

                                            # Probability table
                                            st.subheader("10-Pip Bin Probability Distribution")

                                            # Create DataFrame for display
                                            bin_df = pd.DataFrame(result['bins'])
                                            bin_df['cumulative'] = bin_df['probability'].cumsum()

                                            # Format for display
                                            display_df = pd.DataFrame({
                                                'Pip Range': bin_df['bin_range'],
                                                'Frequency': bin_df['count'],
                                                'Probability (%)': bin_df['probability'].round(2),
                                                'Cumulative (%)': bin_df['cumulative'].round(2)
                                            })

                                            st.dataframe(display_df, use_container_width=True)

                                            # Key insights
                                            st.subheader("Key Insights")

                                            # Find most probable bin
                                            max_prob_bin = max(result['bins'], key=lambda x: x['probability'])

                                            # Calculate low and high volatility probabilities
                                            low_vol_prob = sum(
                                                b['probability'] for b in result['bins'] if b['bin_max'] < 20)
                                            high_vol_prob = sum(
                                                b['probability'] for b in result['bins'] if b['bin_min'] >= 50)

                                            insights = [
                                                f"🎯 **Most Common Range:** {max_prob_bin['bin_range']} pips ({max_prob_bin['probability']:.1f}% probability)",
                                                f"📉 **Low Volatility (<20 pips):** {low_vol_prob:.1f}% of the time",
                                                f"📈 **High Volatility (≥50 pips):** {high_vol_prob:.1f}% of the time",
                                                f"⚖️ **Risk Assessment:** {'Low' if result['mean'] < 20 else 'Medium' if result['mean'] < 40 else 'High'} average volatility hour"
                                            ]

                                            for insight in insights:
                                                st.markdown(insight)

                            # # Trading session analysis
                            # st.subheader("🌍 Trading Session Analysis")
                            #
                            # # Define major trading sessions (UTC hours)
                            # sessions = {
                            #     "Asian Session": list(range(21, 24)) + list(range(0, 7)),  # 21:00 - 07:00 UTC
                            #     "London Session": list(range(7, 12)),  # 07:00 - 12:00 UTC
                            #     "New York Session": list(range(12, 21)),  # 12:00 - 21:00 UTC
                            #     "Asian Time Pocket": list(range(23, 24)) + list(range(0, 1)), # 00:00 to 01:00 UTC (01:00 to 02:00 UK)
                            #     "London Time Pocket": list(range(6, 8)), # 06:00 to 08:00 UTC (07:00 to 09:00 UK)
                            #     "New York Time Pocket": list(range(12, 14))  # 12:00 to 14:00 UTC (13:00 to 15:00 UK)
                            # }
                            #
                            # session_stats = []
                            # for session_name, hours in sessions.items():
                            #     session_hours = [f"{h:02d}:00 UTC" for h in hours if
                            #                      f"{h:02d}:00 UTC" in analysis_results]
                            #     if session_hours:
                            #         session_means = [analysis_results[hour]['mean'] for hour in session_hours]
                            #         session_samples = sum(
                            #             analysis_results[hour]['total_samples'] for hour in session_hours)
                            #
                            #         session_stats.append({
                            #             'Trading Session': session_name,
                            #             'Hours (UTC)': f"{min(hours):02d}:00 - {max(hours):02d}:59",
                            #             'Avg Volatility (pips)': round(np.mean(session_means), 2),
                            #             'Min Volatility (pips)': round(min(session_means), 2),
                            #             'Max Volatility (pips)': round(max(session_means), 2),
                            #             'Total Samples': session_samples,
                            #             'Risk Level': 'Low' if np.mean(session_means) < 20 else 'Medium' if np.mean(
                            #                 session_means) < 40 else 'High'
                            #         })
                            #
                            # if session_stats:
                            #     session_df = pd.DataFrame(session_stats)
                            #     st.dataframe(session_df, use_container_width=True)

                            # Export options
                            st.header("💾 Export Results")

                            # Prepare export data
                            export_data = {}
                            for hour_name, result in analysis_results.items():
                                export_data[f"{hour_name.replace(':', '')}_bins"] = pd.DataFrame(result['bins'])
                                export_data[f"{hour_name.replace(':', '')}_stats"] = pd.DataFrame([{
                                    'metric': k,
                                    'value': v if not isinstance(v, dict) else str(v)
                                } for k, v in result.items() if k != 'bins'])

                            # Create download buttons
                            col1, col2 = st.columns(2)

                            with col1:
                                if st.button("📊 Download Detailed Results (Excel)"):
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                        # Add summary sheet
                                        summary_df.to_excel(writer, sheet_name='Hourly_Summary', index=False)

                                        # Add session analysis if available
                                        if session_stats:
                                            session_df.to_excel(writer, sheet_name='Session_Analysis', index=False)

                                        # Add detailed data for each hour (limit to avoid Excel sheet limit)
                                        for i, (sheet_name, df) in enumerate(
                                                list(export_data.items())[:50]):  # Limit sheets
                                            try:
                                                df.to_excel(writer, sheet_name=sheet_name[:31],
                                                            index=False)  # Excel sheet name limit
                                            except Exception:
                                                continue  # Skip if there's an issue with the sheet

                                    st.download_button(
                                        label="Download Excel File",
                                        data=output.getvalue(),
                                        file_name=f"hourly_volatility_analysis_{st.session_state.selected_pair}_{start_date}_{end_date}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )

                            with col2:
                                if st.button("📋 Download Summary (CSV)"):
                                    csv = summary_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download CSV File",
                                        data=csv,
                                        file_name=f"hourly_volatility_summary_{st.session_state.selected_pair}_{start_date}_{end_date}.csv",
                                        mime="text/csv"
                                    )

                        else:
                            st.warning("No analysis results generated. Please check your date range.")
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

        # Feature overview based on selected analysis mode
        if analysis_mode == "Hourly Volatility Analysis":
            st.subheader("⏰ Hourly Volatility Analysis Features")
            st.markdown("""
            This analysis mode provides:
            - **Hour-by-hour analysis** (0:00-23:59 UTC) of volatility patterns
            - **10-pip bin probability distributions** for each hour
            - **Statistical analysis** (mean, std dev, percentiles) for each hour
            - **Visual overview chart** showing volatility trends throughout the day
            - **Trading session analysis** (Asian, European, US sessions)
            - **Comparative analysis** across all 24 hours
            - **Export capabilities** for further analysis
            - **Flexible hour selection** for detailed analysis

            Perfect for:
            - Identifying optimal trading hours for your strategy
            - Risk assessment for different times of day
            - Understanding daily volatility cycles
            - Position sizing based on time-specific volatility
            - Developing time-based trading rules
            """)
        else:
            st.subheader("🔥 Heatmap Visualization Features")
            st.markdown("""
            This visualization mode provides:
            - **Weekly heatmaps** showing volatility patterns
            - **Color-coded visualization** with pip values displayed
            - **Multiple week comparison** capabilities
            - **Timezone conversion** for local time analysis
            - **Trading day alignment** (Sunday 21:00 UTC start)

            Perfect for:
            - Visual pattern recognition
            - Weekly volatility comparison
            - Quick volatility assessment
            - Time zone specific analysis
            """)


if __name__ == "__main__":
    main()