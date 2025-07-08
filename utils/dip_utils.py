import pandas as pd

def classify_candle(row, small_body_threshold, doji_body_threshold):
    body = abs(row['close'] - row['open'])
    range_ = row['high'] - row['low']

    if range_ == 0:
        return 'flat'

    if body <= doji_body_threshold:
        return 'doji'

    body_ratio = body / range_

    if body_ratio <= small_body_threshold:
        return 'small_bullish' if row['close'] > row['open'] else 'small_bearish'
    elif row['close'] > row['open']:
        return 'bullish'
    elif row['close'] < row['open']:
        return 'bearish'

    return 'flat'

def classify_candles(df, small_body_threshold=0.25, doji_body_threshold=0.0001):
    return df.apply(lambda row: classify_candle(row, small_body_threshold, doji_body_threshold), axis=1)

def detect_dip_patterns(df, candle_types):
    dips = []
    candles = candle_types.tolist()
    i = 0
    dip_id = 0

    while i < len(candles) - 4:
        # Pattern A: bull → ≥2 bears → bull
        if candles[i] == 'bullish' and candles[i+1] == 'bearish' and candles[i+2] == 'bearish':
            j = i + 2
            while j + 1 < len(candles) and candles[j+1] == 'bearish':
                j += 1
            if j + 1 < len(candles) and candles[j+1] == 'bullish':
                dips.append({
                    'id': dip_id,
                    'pattern': 'A',
                    'start_index': i,
                    'end_index': j + 1
                })
                dip_id += 1
                i = j + 1
                continue

        # Pattern B: bull → ≥2 bears → small/doji → bear → bull
        if candles[i] == 'bullish' and candles[i+1] == 'bearish' and candles[i+2] == 'bearish':
            j = i + 2
            while j + 1 < len(candles) and candles[j+1] == 'bearish':
                j += 1
            if (
                j + 3 < len(candles) and
                candles[j+1] in ['small_bullish', 'small_bearish', 'doji'] and
                candles[j+2] == 'bearish' and
                candles[j+3] == 'bullish'
            ):
                dips.append({
                    'id': dip_id,
                    'pattern': 'B',
                    'start_index': i,
                    'end_index': j + 3
                })
                dip_id += 1
                i = j + 3
                continue
        i += 1

    return dips

def validate_dip(df, dip, pip_multiplier, min_pip_drop):
    start, end = dip['start_index'], dip['end_index']
    candles = df.iloc[start:end+1]

    highest_high = candles['high'].max()
    lowest_low = candles.iloc[1:]['low'].min()  # exclude first candle

    pip_drop = (highest_high - lowest_low) * pip_multiplier
    passed = pip_drop >= min_pip_drop

    return {
        'passed': passed,
        'pip_drop': pip_drop,
        'duration': end - start + 1,
        'avg_pip_velocity': pip_drop / (end - start + 1) if passed else 0
    }
