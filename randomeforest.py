"""
evaluate_rushing_model_enhanced.py

Enhanced version that includes 1990s+ data with improved error handling
for older seasons and more robust data collection.
"""

import os
import time
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "."
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}

def fetch_rushing_table(year: int, max_retries: int = 5) -> pd.DataFrame:
    """
    Fetch and clean the full rushing table for a season, with enhanced
    handling for 1990s data structure differences.
    """
    url = f"https://www.pro-football-reference.com/years/{year}/rushing.htm"
    backoff = 5
    
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS)
            if resp.status_code == 429:
                print(f"[{year}] 429 – backing off {backoff}s (attempt {attempt})")
                time.sleep(backoff)
                backoff *= 2
                continue
            elif resp.status_code == 404:
                print(f"[{year}] Data not available (404)")
                return pd.DataFrame()  # Return empty DataFrame for missing years
            
            resp.raise_for_status()
            
            # Try to parse the table
            tables = pd.read_html(resp.text)
            if not tables:
                print(f"[{year}] No tables found")
                return pd.DataFrame()
            
            df = tables[0]
            
            # Handle multi-level columns (common in newer years)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            
            # Remove header rows that appear in data
            df = df[df.iloc[:, 0] != "Rk"].reset_index(drop=True)
            
            # Handle different column names across years
            column_mapping = {
                'Rk': 'Rk', 'Rank': 'Rk',
                'Player': 'Player', 'Name': 'Player',
                'Age': 'Age',
                'Tm': 'Tm', 'Team': 'Tm',
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
            
            # Rename columns to standard format
            df.columns = [column_mapping.get(col, col) for col in df.columns]
            
            # Drop rank column if it exists
            if 'Rk' in df.columns:
                df = df.drop(columns=['Rk'])
            
            # Ensure required columns exist, create with 0s if missing
            required_cols = ['Player', 'Age', 'G', 'GS', 'Att', 'Yds', 'TD', 'Y/A', 'Y/G', 'Fmb']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'Player':
                        print(f"[{year}] Warning: No Player column found")
                        continue
                    df[col] = 0
            
            # Convert numeric columns with enhanced error handling
            numeric_cols = ['Age', 'G', 'GS', 'Att', 'Yds', 'TD', 'Y/A', 'Y/G', 'Fmb']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Filter out rows with missing player names or zero attempts
            df = df.dropna(subset=['Player'])
            df = df[df['Player'].str.strip() != '']
            df = df[df['Att'] > 0].reset_index(drop=True)
            
            print(f"[{year}] Successfully fetched {len(df)} players")
            time.sleep(3)  # Be respectful to the server
            return df
            
        except Exception as e:
            print(f"[{year}] Error on attempt {attempt}: {e}")
            if attempt == max_retries:
                print(f"[{year}] Failed after {max_retries} attempts")
                return pd.DataFrame()
            time.sleep(backoff)
            backoff *= 2
    
    return pd.DataFrame()

def build_extended_historical_training(start_year=1990, end_year=2024):
    """
    Build training data from 1990s+ with enhanced data collection and
    additional features like age and multi-year trends.
    """
    print(f"Collecting data from {start_year} to {end_year-1}...")
    
    # Store all yearly data for trend calculations
    yearly_data = {}
    
    # First pass: collect all available data
    for yr in range(start_year, end_year + 1):
        print(f"Fetching {yr}...")
        df = fetch_rushing_table(yr)
        if not df.empty:
            yearly_data[yr] = df
        else:
            print(f"Skipping {yr} - no data available")
    
    print(f"Successfully collected data for {len(yearly_data)} seasons")
    
    # Second pass: build training pairs with enhanced features
    X_rows, y_rows = [], []
    
    for yr in range(start_year, end_year):
        if yr not in yearly_data or yr + 1 not in yearly_data:
            continue
        
        df_n = yearly_data[yr]
        df_n1 = yearly_data[yr + 1]
        
        # Get top 15 rushers by attempts (more data, still focusing on relevant players)
        top_rushers = df_n.nlargest(15, 'Att').reset_index(drop=True)
        
        for _, player_n in top_rushers.iterrows():
            # Find this player in next year's data
            next_year_data = df_n1[df_n1['Player'] == player_n['Player']]
            
            if next_year_data.empty:
                continue
            
            player_n1 = next_year_data.iloc[0]
            
            # Enhanced features
            features = [
                player_n['Att'],      # Current attempts
                player_n['Yds'],      # Current yards
                player_n['TD'],       # Current TDs
                player_n['Age'],      # Age (crucial for RB predictions)
                player_n['G'],        # Games played
                player_n['Y/A'],      # Efficiency metric
                player_n['Y/G'],      # Yards per game
            ]
            
            # Add 2-year trend if available
            if yr - 1 in yearly_data:
                df_n_minus_1 = yearly_data[yr - 1]
                prev_data = df_n_minus_1[df_n_minus_1['Player'] == player_n['Player']]
                if not prev_data.empty:
                    prev_player = prev_data.iloc[0]
                    # Trend features (current vs previous year)
                    features.extend([
                        player_n['Att'] - prev_player['Att'],    # Attempt trend
                        player_n['Yds'] - prev_player['Yds'],    # Yards trend
                        player_n['Y/A'] - prev_player['Y/A'],    # Efficiency trend
                    ])
                else:
                    features.extend([0, 0, 0])  # No previous year data
            else:
                features.extend([0, 0, 0])  # No previous year available
            
            X_rows.append(features)
            y_rows.append([
                int(player_n1['Att']),
                int(player_n1['Yds']),
                int(player_n1['TD'])
            ])
    
    feature_names = [
        'Att_N', 'Yds_N', 'TD_N', 'Age_N', 'G_N', 'YpA_N', 'YpG_N',
        'Att_trend', 'Yds_trend', 'YpA_trend'
    ]
    
    X = pd.DataFrame(X_rows, columns=feature_names)
    y = pd.DataFrame(y_rows, columns=['Att_N1', 'Yds_N1', 'TD_N1'])
    
    print(f"Built training set with {len(X)} samples across {len(yearly_data)-1} seasons")
    return X, y, yearly_data

def train_enhanced_model(X, y):
    """
    Train both linear regression and random forest models for comparison.
    """
    # Linear regression (your original approach)
    linear_model = MultiOutputRegressor(LinearRegression())
    linear_model.fit(X, y)
    
    # Random Forest (better for non-linear relationships)
    rf_model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
    )
    rf_model.fit(X, y)
    
    return linear_model, rf_model

def enhanced_backtest(models, yearly_data, start_year=2000, end_year=2010):
    """
    Enhanced backtesting with more features and model comparison.
    """
    linear_model, rf_model = models
    results = {'linear': [], 'rf': []}
    
    for yr in range(start_year, end_year):
        if yr not in yearly_data or yr + 1 not in yearly_data:
            continue
            
        df_n = yearly_data[yr]
        df_n1 = yearly_data[yr + 1]
        
        top_rushers = df_n.nlargest(10, 'Att').reset_index(drop=True)
        
        # Build feature matrix for this year
        X_test = []
        test_players = []
        
        for _, player_n in top_rushers.iterrows():
            # Same feature engineering as training
            features = [
                player_n['Att'], player_n['Yds'], player_n['TD'],
                player_n['Age'], player_n['G'], player_n['Y/A'], player_n['Y/G']
            ]
            
            # Add trend features if previous year available
            if yr - 1 in yearly_data:
                df_prev = yearly_data[yr - 1]
                prev_data = df_prev[df_prev['Player'] == player_n['Player']]
                if not prev_data.empty:
                    prev_player = prev_data.iloc[0]
                    features.extend([
                        player_n['Att'] - prev_player['Att'],
                        player_n['Yds'] - prev_player['Yds'],
                        player_n['Y/A'] - prev_player['Y/A'],
                    ])
                else:
                    features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0])
            
            X_test.append(features)
            test_players.append(player_n['Player'])
        
        if not X_test:
            continue
            
        X_test_df = pd.DataFrame(X_test, columns=[
            'Att_N', 'Yds_N', 'TD_N', 'Age_N', 'G_N', 'YpA_N', 'YpG_N',
            'Att_trend', 'Yds_trend', 'YpA_trend'
        ])
        
        # Get predictions from both models
        linear_preds = linear_model.predict(X_test_df)
        rf_preds = rf_model.predict(X_test_df)
        
        # Compare to actual results
        for idx, player in enumerate(test_players):
            actual_data = df_n1[df_n1['Player'] == player]
            if actual_data.empty:
                continue
                
            actual = actual_data.iloc[0]
            
            # Store results for both models
            for model_name, preds in [('linear', linear_preds), ('rf', rf_preds)]:
                pred_att, pred_yds, pred_td = preds[idx]
                
                results[model_name].append({
                    'Season_N': yr,
                    'Player': player,
                    'Age_N': X_test[idx][3],
                    'Att_N': int(X_test[idx][0]),
                    'Yds_N': int(X_test[idx][1]),
                    'TD_N': int(X_test[idx][2]),
                    'Att_pred': round(pred_att),
                    'Yds_pred': round(pred_yds),
                    'TD_pred': round(pred_td),
                    'Att_act': int(actual['Att']),
                    'Yds_act': int(actual['Yds']),
                    'TD_act': int(actual['TD']),
                })
    
    # Calculate metrics for both models
    model_metrics = {}
    
    for model_name in results:
        if not results[model_name]:
            continue
            
        df_preds = pd.DataFrame(results[model_name])
        metrics = {'Model': model_name}
        
        for stat in ['Att', 'Yds', 'TD']:
            mae = mean_absolute_error(df_preds[f'{stat}_act'], df_preds[f'{stat}_pred'])
            mape = (abs(df_preds[f'{stat}_act'] - df_preds[f'{stat}_pred']) /
                   df_preds[f'{stat}_act'].replace(0, 1)).mean() * 100  # Avoid div by 0
            r2 = r2_score(df_preds[f'{stat}_act'], df_preds[f'{stat}_pred'])
            
            metrics[f'{stat}_MAE'] = round(mae, 2)
            metrics[f'{stat}_MAPE'] = round(mape, 2)
            metrics[f'{stat}_R2'] = round(r2, 3)
        
        model_metrics[model_name] = metrics
    
    return results, model_metrics

def main():
    print("Enhanced Rushing Prediction Model with 1990s+ Data")
    print("=" * 55)
    
    print("\n1) Building extended training data from 1990-2023...")
    X, y, yearly_data = build_extended_historical_training(1990, 2024)
    
    if len(X) == 0:
        print("No training data available. Check your data sources.")
        return
    
    print(f"   → {len(X)} training samples with enhanced features")
    print(f"   → Available seasons: {sorted(yearly_data.keys())}")
    
    print("\n2) Training both Linear Regression and Random Forest models...")
    linear_model, rf_model = train_enhanced_model(X, y)
    
    print("\n3) Back-testing on seasons 2000-2009...")
    results, metrics = enhanced_backtest((linear_model, rf_model), yearly_data, 2000, 2010)
    
    print("\nModel Comparison (2001-2010 backtest):")
    print("=" * 50)
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name.upper()} MODEL:")
        for metric, value in model_metrics.items():
            if metric != 'Model':
                print(f"  {metric}: {value}")
    
    # Save detailed results
    for model_name, model_results in results.items():
        if model_results:
            df_results = pd.DataFrame(model_results)
            df_results.to_csv(f"backtest_{model_name}_preds_enhanced.csv", index=False)
            print(f"\nSaved {model_name} predictions → backtest_{model_name}_preds_enhanced.csv")
    
    # Save metrics comparison
    df_metrics = pd.DataFrame(list(metrics.values()))
    df_metrics.to_csv("model_comparison_metrics.csv", index=False)
    print("Saved metrics comparison → model_comparison_metrics.csv")

if __name__ == "__main__":
    main()