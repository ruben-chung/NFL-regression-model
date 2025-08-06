"""
wr_redzone_data_enhanced.py

Enhanced script to get red zone data for Wide Receivers.
Combines web scraping with calculated red zone proxies from existing data.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("nfl_wr_redzone_data")
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

def test_redzone_url_access(year=2023):
    """Test if we can access red zone data URLs."""
    url = f"https://www.pro-football-reference.com/years/{year}/redzone-receiving.htm"
    
    print(f"🔍 Testing red zone URL access for {year}...")
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        print(f"  Status code: {resp.status_code}")
        
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # Look for table
            table = soup.find('table', {'id': 'redzone_receiving'})
            if table:
                print(f"  ✅ Found red zone receiving table!")
                return True
            else:
                print(f"  ❌ No red zone receiving table found")
                
                # Check for any tables
                all_tables = soup.find_all('table')
                print(f"  📊 Found {len(all_tables)} total tables")
                
                for i, tbl in enumerate(all_tables[:3]):
                    table_id = tbl.get('id', 'no-id')
                    print(f"    Table {i+1}: ID = {table_id}")
                
                return False
        else:
            print(f"  ❌ HTTP {resp.status_code} error")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def fetch_redzone_receiving_data_robust(year: int, max_retries: int = 3) -> pd.DataFrame:
    """Robust red zone data fetching with multiple strategies."""
    
    strategies = [
        f"https://www.pro-football-reference.com/years/{year}/redzone-receiving.htm",
        f"https://www.pro-football-reference.com/years/{year}/opp.htm",  # Team stats might have RZ data
    ]
    
    print(f"  Attempting to fetch red zone data for {year}...")
    
    for strategy_idx, url in enumerate(strategies):
        print(f"    Strategy {strategy_idx + 1}: {url.split('/')[-1]}")
        
        for attempt in range(1, max_retries + 1):
            try:
                time.sleep(2)  # Be respectful
                resp = requests.get(url, headers=HEADERS, timeout=15)
                
                if resp.status_code == 429:
                    wait_time = 5 * attempt
                    print(f"      Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif resp.status_code != 200:
                    print(f"      HTTP {resp.status_code}")
                    break
                
                soup = BeautifulSoup(resp.content, 'html.parser')
                
                # Strategy 1: Look for dedicated red zone table
                if 'redzone-receiving' in url:
                    table = soup.find('table', {'id': 'redzone_receiving'})
                    if table:
                        df = parse_redzone_table(table, year)
                        if not df.empty:
                            print(f"      ✅ Success with strategy {strategy_idx + 1}")
                            return df
                
                # Strategy 2: Look for any red zone related tables
                all_tables = soup.find_all('table')
                for table in all_tables:
                    table_id = table.get('id', '').lower()
                    if 'redzone' in table_id or 'red_zone' in table_id:
                        df = parse_redzone_table(table, year)
                        if not df.empty:
                            print(f"      ✅ Found RZ data in table: {table_id}")
                            return df
                
                print(f"      No red zone tables found")
                break
                
            except Exception as e:
                print(f"      Error attempt {attempt}: {e}")
                if attempt == max_retries:
                    break
                time.sleep(3 * attempt)
    
    print(f"    ❌ No red zone data found for {year}")
    return pd.DataFrame()

def parse_redzone_table(table, year):
    """Parse red zone table from BeautifulSoup table element."""
    try:
        # Convert table to pandas
        df = pd.read_html(str(table))[0]
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        
        # Clean data
        df = df[df.iloc[:, 0] != "Rk"].reset_index(drop=True)
        
        # Filter for WRs if position column exists
        if 'Pos' in df.columns:
            df = df[df['Pos'].str.contains('WR|FL|SE', na=False)].copy()
        
        # Rename columns with RZ prefix
        rename_map = {}
        for col in df.columns:
            if col in ['Player', 'Tm', 'Pos']:
                rename_map[col] = col
            elif col != 'Rk':
                rename_map[col] = f'RZ_{col}'
        
        df = df.rename(columns=rename_map)
        
        # Convert numeric columns
        numeric_cols = [col for col in df.columns if col.startswith('RZ_')]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Filter to players with red zone activity
        target_col = next((col for col in df.columns if 'Tgt' in col), None)
        if target_col and target_col in df.columns:
            df = df[df[target_col] > 0]
        
        df['Year'] = year
        
        return df
        
    except Exception as e:
        print(f"      Error parsing table: {e}")
        return pd.DataFrame()

def calculate_redzone_proxies(receiving_df):
    """Calculate red zone proxies from regular receiving stats."""
    print("🔧 Calculating red zone proxies from existing data...")
    
    df = receiving_df.copy()
    
    # Red zone proxy calculations based on TD efficiency and patterns
    
    # 1. TD Rate (higher = more red zone efficiency)
    df['RZ_Proxy_TD_Rate'] = np.where(df['Tgt'] > 0, df['TD'] / df['Tgt'] * 100, 0)
    
    # 2. TD per Reception (red zone conversion efficiency)
    df['RZ_Proxy_TD_Per_Rec'] = np.where(df['Rec'] > 0, df['TD'] / df['Rec'], 0)
    
    # 3. Short Yardage Efficiency (low Y/R but high TDs = red zone work)
    df['RZ_Proxy_Short_TD'] = np.where(
        (df['Y/R'] < 12) & (df['TD'] > 3), 1, 0  # Low Y/R but good TDs
    )
    
    # 4. Estimated Red Zone Targets (based on TD rate patterns)
    # Players with high TD rates likely get more red zone looks
    avg_rz_td_rate = 0.15  # ~15% of RZ targets become TDs (league average)
    df['RZ_Proxy_Est_Targets'] = np.where(
        df['RZ_Proxy_TD_Rate'] > 0,
        df['TD'] / (df['RZ_Proxy_TD_Rate'] / 100) * avg_rz_td_rate,
        0
    )
    
    # 5. Red Zone Role Indicator
    # High TD, moderate targets = red zone specialist
    df['RZ_Proxy_Specialist'] = np.where(
        (df['TD'] >= 6) & (df['Tgt'] < 100) & (df['RZ_Proxy_TD_Rate'] > 10), 1, 0
    )
    
    # 6. Volume vs Efficiency Balance
    df['RZ_Proxy_Volume_Score'] = (
        df['TD'] * 0.4 +  # TDs are key
        (df['RZ_Proxy_TD_Rate'] / 100) * 0.3 +  # Efficiency matters
        np.minimum(df['Tgt'] / 100, 1.0) * 0.3  # Volume helps (capped)
    )
    
    # 7. Projected Red Zone Receptions (very rough estimate)
    df['RZ_Proxy_Est_Receptions'] = np.where(
        df['TD'] > 0,
        df['TD'] + (df['TD'] * 0.5),  # Assume ~1.5 RZ receptions per TD
        0
    )
    
    print(f"  ✅ Calculated {7} red zone proxy metrics")
    print(f"  📊 Players with estimated RZ specialist role: {df['RZ_Proxy_Specialist'].sum()}")
    
    return df

def combine_redzone_and_proxies():
    """Combine scraped red zone data with calculated proxies."""
    print("🔄 Combining scraped and calculated red zone data...")
    
    # Load existing receiving data
    receiving_file = "all_receiving_data.csv"
    if not os.path.exists(receiving_file):
        print("❌ No receiving data found. Run WR data downloader first.")
        return pd.DataFrame()
    
    receiving_df = pd.read_csv(receiving_file)
    print(f"  📊 Loaded {len(receiving_df)} receiving records")
    
    # Calculate proxies for all data
    df_with_proxies = calculate_redzone_proxies(receiving_df)
    
    # Try to load scraped red zone data
    rz_file = DATA_DIR / "all_redzone_receiving_data.csv"
    if rz_file.exists():
        rz_df = pd.read_csv(rz_file)
        print(f"  ✅ Found {len(rz_df)} scraped red zone records")
        
        # Merge scraped data with proxies
        df_combined = df_with_proxies.merge(
            rz_df, on=['Player', 'Year'], how='left'
        )
        print(f"  🔗 Merged data: {len(df_combined)} records")
    else:
        print("  ⚠️ No scraped red zone data found, using proxies only")
        df_combined = df_with_proxies
    
    # Save combined dataset
    output_file = DATA_DIR / "combined_redzone_analysis.csv"
    df_combined.to_csv(output_file, index=False)
    print(f"  💾 Saved combined analysis: {output_file}")
    
    return df_combined

def analyze_redzone_patterns(df):
    """Analyze red zone patterns from the data."""
    print("\n📊 RED ZONE PATTERN ANALYSIS")
    print("=" * 50)
    
    # Filter to relevant players
    df_analysis = df[(df['Tgt'] >= 20) & (df['TD'] > 0)].copy()
    
    print(f"📈 Analysis of {len(df_analysis)} WR seasons (20+ targets, 1+ TD)")
    
    # TD Rate Analysis
    print(f"\n🎯 TD Rate Analysis:")
    print(f"  • Average TD Rate: {df_analysis['RZ_Proxy_TD_Rate'].mean():.1f}%")
    print(f"  • Elite TD Rate (90th percentile): {df_analysis['RZ_Proxy_TD_Rate'].quantile(0.9):.1f}%")
    print(f"  • Poor TD Rate (10th percentile): {df_analysis['RZ_Proxy_TD_Rate'].quantile(0.1):.1f}%")
    
    # Red Zone Specialists
    specialists = df_analysis[df_analysis['RZ_Proxy_Specialist'] == 1]
    if len(specialists) > 0:
        print(f"\n⚡ Red Zone Specialists ({len(specialists)} seasons):")
        top_specialists = specialists.nlargest(10, 'TD')[['Player', 'Year', 'Tgt', 'TD', 'RZ_Proxy_TD_Rate']]
        print(top_specialists.to_string(index=False))
    
    # Efficiency vs Volume
    print(f"\n📊 Efficiency vs Volume Patterns:")
    
    # High efficiency, low volume (red zone specialists)
    high_eff_low_vol = df_analysis[
        (df_analysis['RZ_Proxy_TD_Rate'] > 15) & (df_analysis['Tgt'] < 80)
    ]
    print(f"  • High efficiency, low volume: {len(high_eff_low_vol)} seasons")
    
    # High volume, moderate efficiency (workhorses)
    high_vol_mod_eff = df_analysis[
        (df_analysis['Tgt'] > 120) & (df_analysis['RZ_Proxy_TD_Rate'] > 8)
    ]
    print(f"  • High volume, good efficiency: {len(high_vol_mod_eff)} seasons")
    
    # Year-over-year trends (if multiple years available)
    if df_analysis['Year'].nunique() > 1:
        yearly_trends = df_analysis.groupby('Year').agg({
            'RZ_Proxy_TD_Rate': 'mean',
            'TD': 'mean',
            'RZ_Proxy_Est_Targets': 'mean'
        }).round(2)
        
        print(f"\n📈 Yearly Red Zone Trends:")
        print(yearly_trends.tail(5))
    
    return df_analysis

def download_available_redzone_data(start_year=2018, end_year=2024):
    """Download red zone data for available years."""
    print(f"🏈 ENHANCED RED ZONE DATA COLLECTION")
    print(f"Attempting to collect red zone data from {start_year} to {end_year}")
    print("=" * 60)
    
    # First, test URL access
    if not test_redzone_url_access(2023):
        print("⚠️ Direct red zone URL access seems limited")
        print("   Proceeding with proxy calculations from existing data...")
    
    all_redzone_data = []
    
    for year in range(start_year, end_year + 1):
        print(f"\n📅 Processing {year}:")
        
        # Check if data already exists
        year_file = DATA_DIR / f"redzone_receiving_{year}.csv"
        
        if year_file.exists():
            print(f"  ✅ Red zone data already exists for {year}")
            try:
                df = pd.read_csv(year_file)
                all_redzone_data.append(df)
            except:
                print(f"  ❌ Error loading existing file")
        else:
            # Attempt to fetch new data
            df = fetch_redzone_receiving_data_robust(year)
            if not df.empty:
                df.to_csv(year_file, index=False)
                all_redzone_data.append(df)
                print(f"  💾 Saved {len(df)} records")
            else:
                print(f"  ❌ No data retrieved for {year}")
        
        time.sleep(1)  # Be respectful
    
    # Combine all data
    if all_redzone_data:
        combined_df = pd.concat(all_redzone_data, ignore_index=True)
        output_file = DATA_DIR / "all_redzone_receiving_data.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"\n✅ Combined red zone data: {len(combined_df)} records saved to {output_file}")
    else:
        print(f"\n⚠️ No red zone data successfully scraped")
    
    # Always calculate proxies from existing data
    print(f"\n🔧 Calculating red zone proxies from existing data...")
    combined_analysis = combine_redzone_and_proxies()
    
    if not combined_analysis.empty:
        analyze_redzone_patterns(combined_analysis)
        
        print(f"\n🎉 RED ZONE ANALYSIS COMPLETE!")
        print(f"📁 Files created in: {DATA_DIR}")
        print(f"💡 Key file: combined_redzone_analysis.csv")
        
        return True
    
    return False

def quick_proxy_analysis():
    """Quick analysis using only proxy calculations."""
    print("🚀 QUICK RED ZONE PROXY ANALYSIS")
    print("=" * 40)
    
    combined_df = combine_redzone_and_proxies()
    if not combined_df.empty:
        analyze_redzone_patterns(combined_df)
        return combined_df
    else:
        print("❌ No data available for analysis")
        return pd.DataFrame()

if __name__ == "__main__":
    print("Choose red zone analysis option:")
    print("1. Full scraping attempt + proxy analysis (2018-2024)")
    print("2. Recent years scraping + proxies (2022-2024)")
    print("3. Proxy analysis only (fast, uses existing data)")
    print("4. Test red zone URL access")
    
    choice = input("Enter choice (1/2/3/4): ").strip()
    
    if choice == "1":
        download_available_redzone_data(2018, 2024)
    elif choice == "2":
        download_available_redzone_data(2022, 2024)
    elif choice == "3":
        quick_proxy_analysis()
    elif choice == "4":
        test_redzone_url_access(2023)
        test_redzone_url_access(2022)
        test_redzone_url_access(2020)
    else:
        print("Invalid choice. Running quick proxy analysis...")
        quick_proxy_analysis()