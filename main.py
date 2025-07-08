import pandas as pd
import re
from utils.dip_utils import classify_candles, detect_dip_patterns, validate_dip

# === CONFIG ===
INPUT_FILE = "data/GBPUSD_H4.csv"
SMALL_BODY_THRESHOLD = 0.25

def get_thresholds(symbol):
    if 'JPY' in symbol.upper():
        return {
            'pip_multiplier': 100,
            'doji_body_threshold': 0.01,
            'min_pip_drop': 1.0
        }
    else:
        return {
            'pip_multiplier': 10000,
            'doji_body_threshold': 0.0001,
            'min_pip_drop': 10.0
        }

# === Load Data ===
df = pd.read_csv(INPUT_FILE)

# Convert to datetime (assuming TradingView UTC export)
df['datetime_utc'] = pd.to_datetime(df['time'], unit='s')

# Optional: Extract symbol from filename
symbol_match = re.search(r'([A-Z]{6,7})', INPUT_FILE)
symbol = symbol_match.group(1) if symbol_match else "UNKNOWN"
thresholds = get_thresholds(symbol)

# === Classify candles ===
candle_types = classify_candles(
    df,
    small_body_threshold=SMALL_BODY_THRESHOLD,
    doji_body_threshold=thresholds['doji_body_threshold']
)

# === Detect dips ===
raw_dips = detect_dip_patterns(df, candle_types)

# === Validate dips ===
validated_dips = []
for dip in raw_dips:
    result = validate_dip(
        df,
        dip,
        pip_multiplier=thresholds['pip_multiplier'],
        min_pip_drop=thresholds['min_pip_drop']
    )

    if result['passed']:
        start_time = df.iloc[dip['start_index']]['datetime_utc']
        end_time = df.iloc[dip['end_index']]['datetime_utc']
        session = df.iloc[dip['start_index']].get('session', 'Unknown')

        enriched_dip = {
            **dip,
            **result,
            'start_time': start_time,
            'end_time': end_time,
            'origin_session': session
        }

        validated_dips.append(enriched_dip)

# === Export results ===
output_df = pd.DataFrame(validated_dips)
output_df.to_csv("validated_dips.csv", index=False)
print(f"Saved {len(output_df)} validated dips to 'validated_dips.csv'")
