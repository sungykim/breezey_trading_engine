import pandas as pd

PIP_MULTIPLIERS = {
    'JPY': 100,
    'DEFAULT': 10000
}

def get_pip_multiplier(symbol: str) -> int:
    return PIP_MULTIPLIERS['JPY'] if 'JPY' in symbol else PIP_MULTIPLIERS['DEFAULT']

def classify_candle(row):
    rng = row['high'] - row['low']
    if rng == 0:
        return 'flat'
    if row['close'] > row['open']:
        return 'bullish'
    elif row['close'] < row['open']:
        return 'bearish'
    else:
        return 'CHECK'

def classify_candles(df):
    return df.apply(classify_candle, axis=1)

def detect_dip_patterns(candle_types):
    dips = []
    candles = candle_types.tolist()
    i = 0
    dip_id = 0
    n = len(candles)

    while i < n - 3:
        # Check if starts with bullish
        if candles[i] == 'bullish':
            # Now check how many bearish candles follow starting from i+1
            j = i + 1
            bearish_count = 0

            # Count consecutive bearish candles starting at i+1
            while j < n - 1 and candles[j] == 'bearish':
                bearish_count += 1
                j += 1

            # Need at least 2 bearish candles, and next candle (j) must be bullish to close dip
            if bearish_count >= 2 and j < n and candles[j] == 'bullish':
                dips.append({'id': dip_id, 'start_index': i, 'end_index': j})
                dip_id += 1
                i = j + 1  # Skip past this dip
                continue  # continue the while loop
        i += 1
    return dips

def validate_dip(df, dip, pip_multiplier, min_pip_drop=10):
    start, end = dip['start_index'], dip['end_index']
    candles = df.iloc[start:end+1]

    # Exclude the first candle from low search
    low_candles = candles.iloc[1:]
    highest_high = candles['high'].max()
    lowest_low = low_candles['low'].min()
    pip_drop = (highest_high - lowest_low) * pip_multiplier

    if pip_drop < min_pip_drop:
        return {
            'passed': False,
            'pip_drop': pip_drop,
            'duration': 0,
            'avg_pip_velocity': 0
        }

    # Find the index (relative to df) of high and low
    high_idx_abs = candles['high'].idxmax()
    low_idx_abs = low_candles['low'].idxmin()

    # Convert to relative index within the dip slice
    rel_high_idx = candles.index.get_loc(high_idx_abs)
    rel_low_idx = candles.index.get_loc(low_idx_abs)

    # Compute distance in candles from high to low (must be forward in time)
    if rel_low_idx > rel_high_idx:
        steps = rel_low_idx - rel_high_idx
        avg_velocity = pip_drop / steps if steps > 0 else 0
    else:
        avg_velocity = 0

    # Count only bearish candles in the dip
    bearish_count = (candles['close'] < candles['open']).sum()

    return {
        'passed': True,
        'pip_drop': pip_drop,
        'duration': bearish_count,
        'avg_pip_velocity': avg_velocity
    }