import os
import pandas as pd

RAW_DATA_ROOT = "data/raw_exports"
MERGED_DATA_ROOT = "data"


def merge_and_validate(pair: str, timeframe: str):
    raw_folder = os.path.join(RAW_DATA_ROOT, pair, timeframe)
    merged_folder = os.path.join(MERGED_DATA_ROOT, pair, timeframe)
    os.makedirs(merged_folder, exist_ok=True)
    merged_file = os.path.join(merged_folder, "merged.csv")

    if not os.path.exists(raw_folder):
        print(f"No raw data folder found for {pair} {timeframe}, skipping.")
        return

    all_files = [f for f in os.listdir(raw_folder) if f.endswith(".csv")]
    if not all_files:
        print(f"No CSV files found in {raw_folder}, skipping.")
        return

    dfs = []
    for f in all_files:
        path = os.path.join(raw_folder, f)
        try:
            df = pd.read_csv(path)
            if 'time' not in df.columns or 'open' not in df.columns:
                print(f"Skipping malformed file: {path}")
                continue
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    if not dfs:
        print(f"No valid CSV files to merge for {pair} {timeframe}.")
        return

    merged_df = pd.concat(dfs).drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)

    # Basic time gap validation (assuming consistent timeframe)
    merged_df['time_diff'] = merged_df['time'].diff()
    # 30-minute candles example: 1800 seconds
    expected_diff = 1800 if timeframe.lower() == 'm30' else None

    if expected_diff:
        gaps = merged_df[merged_df['time_diff'] > expected_diff]
        if not gaps.empty:
            print(f"Gaps detected in merged data for {pair} {timeframe} at times:")
            for idx, row in gaps.iterrows():
                prev_time = merged_df.loc[idx - 1, 'time']
                print(f"  Gap from {prev_time} to {row['time']}")

    merged_df = merged_df.drop(columns=['time_diff'])

    merged_df.to_csv(merged_file, index=False)
    print(f"Merged {len(merged_df)} rows for {pair} {timeframe} -> {merged_file}")


if __name__ == "__main__":
    # Example: you can add more pairs/timeframes here
    pairs = ["GBPUSD", "GBPJPY", "GBPAUD"]
    timeframes = ["M30", "H4", "D1"]

    for pair in pairs:
        for timeframe in timeframes:
            merge_and_validate(pair, timeframe)
