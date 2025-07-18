#!/usr/bin/env python3
"""
download_rushing_top10_bulk.py

Fetch NFL rushing stats from Pro-Football-Reference for each season in a given range,
keep only the top 10 players by rushing attempts, and save one CSV per season.

Dependencies:
    pip install pandas requests lxml
"""

import argparse
import os
import requests
import pandas as pd

def fetch_rushing_stats(year: int) -> pd.DataFrame:
    """
    Download and parse the rushing table for the given season year.
    """
    url = f"https://www.pro-football-reference.com/years/{year}/rushing.htm"
    resp = requests.get(url)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)
    df = tables[0]

    # Clean up header rows
    df.columns = df.columns.droplevel(0)
    df = df[df.Rk != 'Rk']  # drop repeating header rows
    df = df.drop(columns=['Rk']).reset_index(drop=True)

    # Convert numeric columns
    numeric_cols = ['Age','G','GS','Att','Yds','TD','Lng','Y/A','Y/G','Fmb']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def save_top10_by_attempts(df: pd.DataFrame, year: int, out_dir: str):
    """
    Sort the DataFrame by 'Att' descending, take top 10, and save to CSV.
    """
    top10 = df.sort_values('Att', ascending=False).head(10)
    filename = os.path.join(out_dir, f"top10_rushing_{year}.csv")
    top10.to_csv(filename, index=False)
    print(f"Saved top 10 for {year} ➜ {filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Download NFL rushing top-10 by attempts for each season in a year range."
    )
    parser.add_argument(
        "--start-year", type=int, default=2000,
        help="First season year (inclusive). Default: 2010"
    )
    parser.add_argument(
        "--end-year", type=int, default=2010,
        help="Last season year (inclusive). Default: 2024"
    )
    parser.add_argument(
        "--out-dir", type=str, default=".",
        help="Directory to save CSV files. Default: current directory"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for year in range(args.start_year, args.end_year + 1):
        print(f"\nProcessing season {year}...")
        try:
            df = fetch_rushing_stats(year)
            save_top10_by_attempts(df, year, args.out_dir)
        except Exception as e:
            print(f"  ❌ Failed for {year}: {e}")

if __name__ == "__main__":
    main()
