"""
data_downloader.py

Separate script to download and cache NFL rushing data.
Run this once to build your dataset, then use the ML script.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("nfl_data")
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}

def fetch_rushing_data(year: int, max_retries: int = 3) -> pd.DataFrame:
    """Fetch rushing data for a single year."""
    url = f"https://www.pro-football-reference.com/years/{year}/rushing.htm"
    backoff = 3
    
    print(f"  Fetching rushing data for {year}...")
    
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 429:
                print(f"    Rate limited, waiting {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            elif resp.status_code == 404:
                print(f"    No data found for {year}")
                return pd.DataFrame()
            
            resp.raise_for_status()
            tables = pd.read_html(resp.text)
            if not tables:
                return pd.DataFrame()
            
            df = tables[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            
            # Clean the data
            df = df[df.iloc[:, 0] != "Rk"].reset_index(drop=True)
            
            # Standardize column names
            column_mapping = {
                'Rk': 'Rk', 'Rank': 'Rk',
                'Player': 'Player', 'Name': 'Player',
                'Age': 'Age',
                'Tm': 'Tm', 'Team': 'Tm',
                'Pos': 'Pos', 'Position': 'Pos',
                'G': 'G', 'Games': 'G',
                'GS': 'GS', 'Starts': 'GS',
                'Att': 'Att', 'Rush Att': 'Att', 'Rushing Att': 'Att',
                'Yds': 'Yds', 'Rush Yds': 'Yds', 'Rushing Yds': 'Yds',
                'TD': 'TD', 'Rush TD': 'TD', 'Rushing TD': 'TD',
                'Y/A': 'Y/A', 'Yds/Att': 'Y/A',
                'Y/G': 'Y/G', 'Yds/G': 'Y/G',
                'Lng': 'Lng', 'Long': 'Lng',
                'Fmb': 'Fmb', 'Fumbles': 'Fmb'
            }
            
            df.columns = [column_mapping.get(col, col) for col in df.columns]
            
            # Remove rank column if it exists
            if 'Rk' in df.columns:
                df = df.drop(columns=['Rk'])
            
            # Ensure required columns exist
            required_cols = ['Player', 'Age', 'Tm', 'G', 'GS', 'Att', 'Yds', 'TD', 'Y/A', 'Y/G', 'Fmb']
            for col in required_cols:
                if col not in df.columns:
                    if col in ['Player', 'Tm']:
                        continue
                    df[col] = 0
            
            # Convert to numeric
            numeric_cols = ['Age', 'G', 'GS', 'Att', 'Yds', 'TD', 'Y/A', 'Y/G', 'Fmb']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Clean and filter
            df = df.dropna(subset=['Player'])
            df = df[df['Player'].str.strip() != '']
            df = df[df['Att'] > 0].reset_index(drop=True)
            
            # Add year column
            df['Year'] = year
            
            print(f"    ✅ Found {len(df)} rushing players")
            return df
            
        except Exception as e:
            print(f"    Error attempt {attempt}: {e}")
            if attempt == max_retries:
                return pd.DataFrame()
            time.sleep(backoff)
            backoff *= 2
    
    return pd.DataFrame()

def fetch_receiving_data(year: int, max_retries: int = 3) -> pd.DataFrame:
    """Fetch receiving data for RBs."""
    url = f"https://www.pro-football-reference.com/years/{year}/receiving.htm"
    backoff = 3
    
    print(f"  Fetching receiving data for {year}...")
    
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 429:
                print(f"    Rate limited, waiting {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            elif resp.status_code != 200:
                return pd.DataFrame()
            
            tables = pd.read_html(resp.text)
            if not tables:
                return pd.DataFrame()
            
            df = tables[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            
            df = df[df.iloc[:, 0] != "Rk"].reset_index(drop=True)
            
            # Focus on RBs only
            if 'Pos' in df.columns:
                df = df[df['Pos'].str.contains('RB|FB', na=False)].copy()
            
            # Get receiving columns
            rec_cols = ['Player', 'Tm', 'Tgt', 'Rec', 'Yds', 'TD', 'Y/R', 'Y/Tgt']
            available_cols = [col for col in rec_cols if col in df.columns]
            
            if available_cols and len(df) > 0:
                rec_df = df[available_cols].copy()
                
                # Rename to avoid conflicts with rushing stats
                rename_map = {
                    'Tgt': 'Rec_Tgt',
                    'Rec': 'Rec_Rec', 
                    'Yds': 'Rec_Yds',
                    'TD': 'Rec_TD',
                    'Y/R': 'Rec_Y/R',
                    'Y/Tgt': 'Rec_Y/Tgt'
                }
                rec_df = rec_df.rename(columns=rename_map)
                
                # Convert to numeric
                numeric_cols = [col for col in rec_df.columns if col not in ['Player', 'Tm']]
                for col in numeric_cols:
                    rec_df[col] = pd.to_numeric(rec_df[col], errors='coerce').fillna(0)
                
                # Add year
                rec_df['Year'] = year
                
                print(f"    ✅ Found {len(rec_df)} RB receiving stats")
                return rec_df
            
        except Exception as e:
            if attempt == max_retries:
                return pd.DataFrame()
            time.sleep(backoff)
            backoff *= 2
    
    return pd.DataFrame()

def fetch_team_stats(year: int, max_retries: int = 3) -> pd.DataFrame:
    """Fetch team stats for context."""
    url = f"https://www.pro-football-reference.com/years/{year}/opp.htm"
    backoff = 3
    
    print(f"  Fetching team stats for {year}...")
    
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 429:
                time.sleep(backoff)
                backoff *= 2
                continue
            elif resp.status_code != 200:
                return pd.DataFrame()
            
            tables = pd.read_html(resp.text)
            if not tables:
                return pd.DataFrame()
            
            df = tables[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            
            df = df[df.iloc[:, 0] != "Tm"].reset_index(drop=True)
            
            # Get relevant team stats
            relevant_cols = ['Tm', 'G', 'PF', 'PA', 'TOT', 'Ply', 'Y/P']
            available_cols = [col for col in relevant_cols if col in df.columns]
            
            if available_cols:
                team_df = df[available_cols].copy()
                numeric_cols = [col for col in available_cols if col != 'Tm']
                for col in numeric_cols:
                    team_df[col] = pd.to_numeric(team_df[col], errors='coerce')
                
                # Add year
                team_df['Year'] = year
                
                print(f"    ✅ Found {len(team_df)} team stats")
                return team_df
                
        except Exception as e:
            if attempt == max_retries:
                return pd.DataFrame()
            time.sleep(backoff)
            backoff *= 2
    
    return pd.DataFrame()

def download_all_data(start_year=1990, end_year=2024):
    """Download all data and save to CSV files."""
    print(f"🏈 NFL DATA DOWNLOADER")
    print(f"Downloading data from {start_year} to {end_year}")
    print("=" * 50)
    
    all_rushing = []
    all_receiving = []
    all_team_stats = []
    
    for year in range(start_year, end_year + 1):
        print(f"\n📅 Processing {year}:")
        
        # Check if data already exists
        rushing_file = DATA_DIR / f"rushing_{year}.csv"
        receiving_file = DATA_DIR / f"receiving_{year}.csv"
        team_file = DATA_DIR / f"teams_{year}.csv"
        
        # Fetch rushing data
        if rushing_file.exists():
            print(f"  ✅ Rushing data already exists for {year}")
            rushing_df = pd.read_csv(rushing_file)
        else:
            rushing_df = fetch_rushing_data(year)
            if not rushing_df.empty:
                rushing_df.to_csv(rushing_file, index=False)
                time.sleep(2)  # Be nice to the server
        
        if not rushing_df.empty:
            all_rushing.append(rushing_df)
        
        # Fetch receiving data
        if receiving_file.exists():
            print(f"  ✅ Receiving data already exists for {year}")
            receiving_df = pd.read_csv(receiving_file)
        else:
            receiving_df = fetch_receiving_data(year)
            if not receiving_df.empty:
                receiving_df.to_csv(receiving_file, index=False)
                time.sleep(2)
        
        if not receiving_df.empty:
            all_receiving.append(receiving_df)
        
        # Fetch team data
        if team_file.exists():
            print(f"  ✅ Team data already exists for {year}")
            team_df = pd.read_csv(team_file)
        else:
            team_df = fetch_team_stats(year)
            if not team_df.empty:
                team_df.to_csv(team_file, index=False)
                time.sleep(2)
        
        if not team_df.empty:
            all_team_stats.append(team_df)
    
    # Combine all data
    print(f"\n📊 Combining all data...")
    
    if all_rushing:
        combined_rushing = pd.concat(all_rushing, ignore_index=True)
        combined_file = DATA_DIR / "all_rushing_data.csv"
        combined_rushing.to_csv(combined_file, index=False)
        print(f"  ✅ Saved {len(combined_rushing)} rushing records → {combined_file}")
    
    if all_receiving:
        combined_receiving = pd.concat(all_receiving, ignore_index=True)
        receiving_file = DATA_DIR / "all_receiving_data.csv"
        combined_receiving.to_csv(receiving_file, index=False)
        print(f"  ✅ Saved {len(combined_receiving)} receiving records → {receiving_file}")
    
    if all_team_stats:
        combined_teams = pd.concat(all_team_stats, ignore_index=True)
        teams_file = DATA_DIR / "all_team_stats.csv"
        combined_teams.to_csv(teams_file, index=False)
        print(f"  ✅ Saved {len(combined_teams)} team records → {teams_file}")
    
    print(f"\n🎉 Data download complete!")
    print(f"📁 All files saved in: {DATA_DIR}")
    
    return len(all_rushing) > 0

def quick_download_recent(start_year=2020):
    """Quick download of just recent years for testing."""
    return download_all_data(start_year, 2024)

if __name__ == "__main__":
    print("Choose download option:")
    print("1. Full dataset (1990-2024) - Takes ~30-45 minutes")
    print("2. Recent years only (2020-2024) - Takes ~5 minutes")
    print("3. Quick test (2022-2024) - Takes ~2 minutes")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        download_all_data(1990, 2024)
    elif choice == "2":
        download_all_data(2020, 2024)
    elif choice == "3":
        download_all_data(2022, 2024)
    else:
        print("Invalid choice. Running quick test...")
        download_all_data(2022, 2024)