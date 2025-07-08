# time_utils.py
import pandas as pd
import pytz

def convert_to_datetime(df, time_col, time_format='auto', unit='s', target_tz=None):
    series = df[time_col]

    if time_format == 'auto':
        if pd.api.types.is_numeric_dtype(series):
            time_format = 'unix'
        else:
            time_format = 'iso'

    if time_format == 'unix':
        dt_utc = pd.to_datetime(series, unit=unit, utc=True)
    elif time_format == 'iso':
        dt_utc = pd.to_datetime(series, utc=True)
    else:
        raise ValueError("time_format must be 'auto', 'unix', or 'iso'")

    df['datetime_utc'] = dt_utc

    if target_tz:
        df['datetime_local'] = df['datetime_utc'].dt.tz_convert(target_tz)

    return df


def get_session(timestamp):
    london_tz = pytz.timezone('Europe/London')
    ny_tz = pytz.timezone('America/New_York')

    london_time = timestamp.astimezone(london_tz)
    ny_time = timestamp.astimezone(ny_tz)
    hour_utc = timestamp.hour

    if 0 <= hour_utc < 9:
        return 'Tokyo'
    if 7 <= london_time.hour < 16:
        return 'London'
    if 8 <= ny_time.hour < 17:
        return 'New York'

    return 'Off-hours'


def assign_sessions(df):
    df['session'] = df['datetime_utc'].apply(get_session)
    return df
