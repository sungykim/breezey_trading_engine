import os
import pandas as pd
from utils.dip_utils import classify_candles, detect_dip_patterns, validate_dip, get_pip_multiplier
from utils.session_utils import get_origin_session

# === Config ===
DATA_ROOT = "data"
RESULTS_ROOT = "results"

def process_merged_files():
    for pair in os.listdir(DATA_ROOT):
        pair_path = os.path.join(DATA_ROOT, pair)
        if not os.path.isdir(pair_path) or pair.startswith('.'):
            continue

        for timeframe in os.listdir(pair_path):
            timeframe_path = os.path.join(pair_path, timeframe)
            if not os.path.isdir(timeframe_path) or timeframe.startswith('.'):
                continue

            # File like GBPAUD_D1_merged.csv
            merged_filename = f"{pair}_{timeframe}_merged.csv"
            merged_file_path = os.path.join(timeframe_path, merged_filename)

            if not os.path.exists(merged_file_path):
                print(f"❌ No merged file found for {pair} {timeframe}, skipping.")
                continue

            print(f"🔍 Processing {merged_file_path}...")
            df = pd.read_csv(merged_file_path)
            df['datetime_utc'] = pd.to_datetime(df['time'], unit='s')

            pip_multiplier = get_pip_multiplier(pair.upper())

            candle_types = classify_candles(df)
            dips = detect_dip_patterns(candle_types)

            validated_dips = []
            for dip in dips:
                validation = validate_dip(df, dip, pip_multiplier)
                if not validation['passed']:
                    continue

                start_idx = dip['start_index']
                end_idx = dip['end_index']

                dip_data = {
                    'pair': pair,
                    'timeframe': timeframe,
                    'dip_id': dip['id'],
                    'start_time': df.iloc[start_idx]['datetime_utc'],
                    'end_time': df.iloc[end_idx]['datetime_utc'],
                    'origin_session': get_origin_session(df.iloc[start_idx]['datetime_utc']),
                    **validation
                }

                validated_dips.append(dip_data)

            if validated_dips:
                results_df = pd.DataFrame(validated_dips)

                # Create per-pair/timeframe results folder
                results_folder = os.path.join(RESULTS_ROOT, pair, timeframe)
                os.makedirs(results_folder, exist_ok=True)

                output_file = os.path.join(results_folder, f"{pair}_{timeframe}_validated_dips.csv")
                results_df.to_csv(output_file, index=False)

                print(f"✅ Saved {len(results_df)} validated dips to {output_file}")
            else:
                print(f"⚠️ No validated dips for {pair} {timeframe}")

if __name__ == "__main__":
    process_merged_files()
