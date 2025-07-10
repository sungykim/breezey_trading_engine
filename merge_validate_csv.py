import os
import shutil
import pandas as pd

DATA_ROOT = "data"  # Root folder for both merged and raw data

def merge_and_validate(pair: str, timeframe: str):
    raw_folder = os.path.join(DATA_ROOT, pair, timeframe, "raw_exports")
    processed_folder = os.path.join(raw_folder, "processed")  # Where we move processed files
    merged_folder = os.path.join(DATA_ROOT, pair, timeframe)
    os.makedirs(processed_folder, exist_ok=True)
    os.makedirs(merged_folder, exist_ok=True)

    merged_file = os.path.join(merged_folder, f"{pair}_{timeframe}_merged.csv")

    if not os.path.exists(raw_folder):
        print(f"No raw data folder found for {pair} {timeframe}, skipping.")
        return

    all_files = [f for f in os.listdir(raw_folder) if f.endswith(".csv") and not f.startswith("processed")]
    if not all_files:
        print(f"No CSV files found in {raw_folder}, skipping.")
        return

    dfs = []
    processed_files = []

    for f in all_files:
        path = os.path.join(raw_folder, f)
        try:
            df = pd.read_csv(path)
            if 'time' not in df.columns or 'open' not in df.columns:
                print(f"Skipping malformed file: {path}")
                continue
            dfs.append(df)
            processed_files.append(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    if not dfs:
        print(f"No valid CSV files to merge for {pair} {timeframe}.")
        return

    new_data = pd.concat(dfs).drop_duplicates(subset=['time'])

    if os.path.exists(merged_file):
        existing_data = pd.read_csv(merged_file)
        combined = pd.concat([existing_data, new_data])
        combined = combined.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
        new_rows = len(combined) - len(existing_data)
    else:
        combined = new_data.sort_values('time').reset_index(drop=True)
        new_rows = len(combined)

    combined['time_diff'] = combined['time'].diff()
    expected_diff = 1800 if timeframe.lower() == 'm30' else None

    if expected_diff:
        gaps = combined[combined['time_diff'] > expected_diff]
        if not gaps.empty:
            print(f"Gaps detected in merged data for {pair} {timeframe} at times:")
            for idx, row in gaps.iterrows():
                prev_time = combined.loc[idx - 1, 'time']
                print(f"  Gap from {prev_time} to {row['time']}")

    combined = combined.drop(columns=['time_diff'])
    combined.to_csv(merged_file, index=False)
    print(f"✅ Merged total: {len(combined)} rows for {pair} {timeframe} ({new_rows} new)")

    # Move processed files to /processed/
    for fpath in processed_files:
        fname = os.path.basename(fpath)
        dest = os.path.join(processed_folder, fname)
        shutil.move(fpath, dest)
        print(f"📦 Moved processed file to: {dest}")


if __name__ == "__main__":
    pairs = ["GBPUSD", "GBPJPY", "GBPAUD"]
    timeframes = ["M30", "H4", "D1"]

    for pair in pairs:
        for timeframe in timeframes:
            merge_and_validate(pair, timeframe)
