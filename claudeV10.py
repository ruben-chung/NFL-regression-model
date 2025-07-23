"""
rushing_predictor.py

Fast ML analysis using pre-downloaded CSV data with aggregated rushing/receiving stats.
Optimized for the specific data structure provided.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.neural_network import MLPRegressor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("nfl_data")

def load_and_merge_data():
    """Load and intelligently merge rushing and receiving data."""
    print("📂 Loading and merging data files...")
    
    # Load rushing data
    rushing_file = DATA_DIR / "all_rushing_data.csv"
    if not rushing_file.exists():
        print(f"  ❌ No rushing data found at {rushing_file}")
        return None
    
    rushing_df = pd.read_csv(rushing_file)
    print(f"  ✅ Loaded {len(rushing_df)} rushing records")
    print(f"  Rushing columns: {list(rushing_df.columns)}")
    
    # Load receiving data
    receiving_file = DATA_DIR / "all_receiving_data.csv"
    receiving_df = None
    if receiving_file.exists():
        receiving_df = pd.read_csv(receiving_file)
        print(f"  ✅ Loaded {len(receiving_df)} receiving records")
        print(f"  Receiving columns: {list(receiving_df.columns)}")
        
        # Merge rushing and receiving data on Player and Year
        print("  🔄 Merging rushing and receiving data...")
        combined_df = pd.merge(
            rushing_df, 
            receiving_df, 
            on=['Player', 'Year'], 
            how='left'
        )
        print(f"  ✅ Merged data: {len(combined_df)} records")
        
    else:
        print("  ⚠️ No receiving data found, using rushing data only")
        combined_df = rushing_df.copy()
        # Add empty receiving columns
        combined_df['Rec_Rec'] = 0
        combined_df['Rec_Yds'] = 0
        combined_df['Rec_TD'] = 0
        combined_df['Rec_Y/R'] = 0
        combined_df['Rec_Tgt'] = 0
        combined_df['Rec_Y/Tgt'] = 0
    
    # Load team data
    teams_file = DATA_DIR / "all_team_stats.csv"
    team_data = None
    if teams_file.exists():
        team_data = pd.read_csv(teams_file)
        print(f"  ✅ Loaded {len(team_data)} team records")
        print(f"  Team columns: {list(team_data.columns)}")
    
    # Fill missing values
    combined_df = combined_df.fillna(0)
    
    return {
        'combined': combined_df,
        'teams': team_data
    }

def calculate_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive features using the exact column names from your data."""
    df_enhanced = df.copy()
    
    # Fill NaN values first
    df_enhanced = df_enhanced.fillna(0)
    
    # Basic rushing efficiency metrics (using exact column names)
    df_enhanced['Att_per_Game'] = np.where(df_enhanced['G'] > 0, 
                                          df_enhanced['Att'] / df_enhanced['G'], 0)
    df_enhanced['TD_per_Att'] = np.where(df_enhanced['Att'] > 0, 
                                        df_enhanced['TD'] / df_enhanced['Att'], 0)
    df_enhanced['Fmb_Rate'] = np.where(df_enhanced['Att'] > 0, 
                                      df_enhanced['Fmb'] / df_enhanced['Att'], 0)
    df_enhanced['Games_Started_Pct'] = np.where(df_enhanced['G'] > 0, 
                                               df_enhanced['GS'] / df_enhanced['G'], 0)
    
    # Receiving efficiency metrics (using exact column names)
    df_enhanced['Rec_per_Game'] = np.where(df_enhanced['G'] > 0, 
                                          df_enhanced['Rec_Rec'] / df_enhanced['G'], 0)
    df_enhanced['Rec_TD_per_Rec'] = np.where(df_enhanced['Rec_Rec'] > 0, 
                                            df_enhanced['Rec_TD'] / df_enhanced['Rec_Rec'], 0)
    df_enhanced['Rec_Yds_per_Game'] = np.where(df_enhanced['G'] > 0, 
                                              df_enhanced['Rec_Yds'] / df_enhanced['G'], 0)
    df_enhanced['Catch_Rate'] = np.where(df_enhanced['Rec_Tgt'] > 0, 
                                        df_enhanced['Rec_Rec'] / df_enhanced['Rec_Tgt'], 0)
    
    # Combined offensive metrics
    df_enhanced['Total_Yds'] = df_enhanced['Yds'] + df_enhanced['Rec_Yds']
    df_enhanced['Total_TD'] = df_enhanced['TD'] + df_enhanced['Rec_TD']
    df_enhanced['Total_Touches'] = df_enhanced['Att'] + df_enhanced['Rec_Rec']
    df_enhanced['Total_Yds_per_Touch'] = np.where(df_enhanced['Total_Touches'] > 0, 
                                                 df_enhanced['Total_Yds'] / df_enhanced['Total_Touches'], 0)
    df_enhanced['Total_Yds_per_Game'] = np.where(df_enhanced['G'] > 0, 
                                                df_enhanced['Total_Yds'] / df_enhanced['G'], 0)
    
    # Usage and workload categories
    df_enhanced['High_Usage_Rush'] = (df_enhanced['Att'] >= 200).astype(int)
    df_enhanced['Workhorse_Rush'] = (df_enhanced['Att'] >= 300).astype(int)
    df_enhanced['High_Usage_Rec'] = (df_enhanced['Rec_Rec'] >= 50).astype(int)
    df_enhanced['Dual_Threat'] = ((df_enhanced['Att'] >= 100) & (df_enhanced['Rec_Rec'] >= 30)).astype(int)
    df_enhanced['High_Touch_Total'] = (df_enhanced['Total_Touches'] >= 250).astype(int)
    df_enhanced['Goal_Line_Back'] = (df_enhanced['TD'] >= 10).astype(int)
    
    # Age features
    df_enhanced['Age_Squared'] = df_enhanced['Age'] ** 2
    df_enhanced['Prime_Age'] = ((df_enhanced['Age'] >= 24) & (df_enhanced['Age'] <= 28)).astype(int)
    df_enhanced['Rookie'] = (df_enhanced['Age'] <= 22).astype(int)
    df_enhanced['Veteran'] = (df_enhanced['Age'] >= 30).astype(int)
    
    # Versatility and role metrics
    df_enhanced['Rush_Rec_Ratio'] = np.where(df_enhanced['Rec_Rec'] > 0, 
                                            df_enhanced['Att'] / df_enhanced['Rec_Rec'], 
                                            df_enhanced['Att'])
    df_enhanced['Receiving_Dependence'] = np.where(df_enhanced['Total_Yds'] > 0, 
                                                  df_enhanced['Rec_Yds'] / df_enhanced['Total_Yds'], 0)
    
    # Efficiency vs Volume balance
    df_enhanced['Efficiency_Score'] = df_enhanced['Y/A'] * df_enhanced['Rec_Y/R'] * 0.1
    df_enhanced['Volume_Score'] = (df_enhanced['Att'] + df_enhanced['Rec_Rec']) * 0.01
    
    # Success rate features (using your Succ% column)
    df_enhanced['Success_Rate_Adj'] = df_enhanced['Succ%'] * 0.01  # Convert percentage to decimal
    df_enhanced['First_Down_Rate'] = np.where(df_enhanced['Att'] > 0, 
                                             df_enhanced['1D'] / df_enhanced['Att'], 0)
    
    return df_enhanced

def calculate_yearly_trends(df):
    """Calculate year-over-year trends for comprehensive metrics."""
    print("  Calculating comprehensive year-over-year trends...")
    
    # Group by player and calculate shifts
    player_groups = df.groupby('Player')
    trend_features = []
    
    for player, group in player_groups:
        group = group.sort_values('Year')
        
        for i in range(1, len(group)):
            current = group.iloc[i]
            previous = group.iloc[i-1]
            
            # Only if consecutive years
            if current['Year'] - previous['Year'] == 1:
                trend_row = current.copy()
                
                # Rushing trends
                trend_row['Att_Change_1yr'] = current['Att'] - previous['Att']
                trend_row['Yds_Change_1yr'] = current['Yds'] - previous['Yds']
                trend_row['YpA_Change_1yr'] = current['Y/A'] - previous['Y/A']
                trend_row['TD_Change_1yr'] = current['TD'] - previous['TD']
                
                # Receiving trends
                trend_row['Rec_Change_1yr'] = current['Rec_Rec'] - previous['Rec_Rec']
                trend_row['Rec_Yds_Change_1yr'] = current['Rec_Yds'] - previous['Rec_Yds']
                trend_row['Rec_TD_Change_1yr'] = current['Rec_TD'] - previous['Rec_TD']
                
                # Combined trends
                trend_row['Total_Yds_Change_1yr'] = current['Total_Yds'] - previous['Total_Yds']
                trend_row['Total_Touches_Change_1yr'] = current['Total_Touches'] - previous['Total_Touches']
                trend_row['Total_TD_Change_1yr'] = current['Total_TD'] - previous['Total_TD']
                
                # Age progression
                trend_row['Age_Change'] = current['Age'] - previous['Age']
                trend_row['Career_Progression'] = i  # Years in dataset
                
                # Two year trends if available
                if i >= 2:
                    prev_prev = group.iloc[i-2]
                    if previous['Year'] - prev_prev['Year'] == 1:
                        trend_row['Att_Change_2yr'] = current['Att'] - prev_prev['Att']
                        trend_row['Total_Yds_Change_2yr'] = current['Total_Yds'] - prev_prev['Total_Yds']
                        trend_row['YpA_Trend_2yr'] = (current['Y/A'] + previous['Y/A']) / 2 - prev_prev['Y/A']
                
                trend_features.append(trend_row)
    
    if trend_features:
        trend_df = pd.DataFrame(trend_features)
        
        # Fill missing trend columns
        trend_cols = ['Att_Change_1yr', 'Yds_Change_1yr', 'YpA_Change_1yr', 'TD_Change_1yr',
                     'Rec_Change_1yr', 'Rec_Yds_Change_1yr', 'Rec_TD_Change_1yr',
                     'Total_Yds_Change_1yr', 'Total_Touches_Change_1yr', 'Total_TD_Change_1yr',
                     'Age_Change', 'Career_Progression', 'Att_Change_2yr', 'Total_Yds_Change_2yr',
                     'YpA_Trend_2yr']
        
        for col in trend_cols:
            if col not in trend_df.columns:
                trend_df[col] = 0
            else:
                trend_df[col] = trend_df[col].fillna(0)
        
        print(f"    ✅ Added comprehensive trends for {len(trend_df)} player-seasons")
        return trend_df
    
    return df

def build_comprehensive_training_data(data_dict, min_year=1990, max_year=2023):
    """Build comprehensive training data from merged rushing/receiving data."""
    print(f"🔄 Building comprehensive training data ({min_year}-{max_year})...")
    
    combined_df = data_dict['combined']
    
    # Filter years
    combined_df = combined_df[
        (combined_df['Year'] >= min_year) & 
        (combined_df['Year'] <= max_year)
    ].copy()
    
    # Add comprehensive features
    enhanced_df = calculate_comprehensive_features(combined_df)
    
    # Add trends if we have multiple years
    if enhanced_df['Year'].nunique() > 1:
        enhanced_df = calculate_yearly_trends(enhanced_df)
    
    # Merge with team stats if available
    if 'teams' in data_dict and data_dict['teams'] is not None:
        teams_df = data_dict['teams']
        teams_df = teams_df[
            (teams_df['Year'] >= min_year) & 
            (teams_df['Year'] <= max_year)
        ]
        
        enhanced_df = pd.merge(
            enhanced_df,
            teams_df,
            on=['Tm', 'Year'],
            how='left',
            suffixes=('', '_Team')
        )
    
    # Fill missing values
    enhanced_df = enhanced_df.fillna(0)
    
    print("  Creating year-to-year training pairs...")
    
    X_rows, y_rows = [], []
    
    # Group by player and create consecutive year pairs
    player_groups = enhanced_df.groupby('Player')
    
    for player, group in player_groups:
        group = group.sort_values('Year')
        
        for i in range(len(group) - 1):
            current = group.iloc[i]
            next_year = group.iloc[i + 1]
            
            # Only use consecutive years and players with decent usage
            if (next_year['Year'] - current['Year'] == 1 and 
                current['Att'] >= 30):  # Lower threshold for more data
                
                # Comprehensive feature vector using exact column names
                features = [
                    # Core rushing stats (9 features)
                    current.get('Att', 0), current.get('Yds', 0), current.get('TD', 0),
                    current.get('Age', 25), current.get('G', 0), current.get('GS', 0),
                    current.get('Y/A', 0), current.get('Y/G', 0), current.get('Fmb', 0),
                    
                    # Core receiving stats (6 features)
                    current.get('Rec_Rec', 0), current.get('Rec_Yds', 0), current.get('Rec_TD', 0),
                    current.get('Rec_Y/R', 0), current.get('Rec_Tgt', 0), current.get('Rec_Y/Tgt', 0),
                    
                    # Combined metrics (5 features)
                    current.get('Total_Yds', 0), current.get('Total_TD', 0), 
                    current.get('Total_Touches', 0), current.get('Total_Yds_per_Touch', 0),
                    current.get('Total_Yds_per_Game', 0),
                    
                    # Efficiency metrics (8 features)
                    current.get('Att_per_Game', 0), current.get('TD_per_Att', 0), 
                    current.get('Fmb_Rate', 0), current.get('Games_Started_Pct', 0),
                    current.get('Rec_per_Game', 0), current.get('Rec_TD_per_Rec', 0),
                    current.get('Rec_Yds_per_Game', 0), current.get('Catch_Rate', 0),
                    
                    # Usage and role indicators (6 features)
                    current.get('High_Usage_Rush', 0), current.get('Workhorse_Rush', 0),
                    current.get('High_Usage_Rec', 0), current.get('Dual_Threat', 0),
                    current.get('High_Touch_Total', 0), current.get('Goal_Line_Back', 0),
                    
                    # Versatility metrics (4 features)
                    current.get('Rush_Rec_Ratio', 0), current.get('Receiving_Dependence', 0),
                    current.get('Efficiency_Score', 0), current.get('Volume_Score', 0),
                    
                    # Age features (4 features)
                    current.get('Age_Squared', 0), current.get('Prime_Age', 0), 
                    current.get('Rookie', 0), current.get('Veteran', 0),
                    
                    # Success metrics (2 features)
                    current.get('Success_Rate_Adj', 0), current.get('First_Down_Rate', 0),
                    
                    # Team context (3 features - if available)
                    current.get('PA', current.get('PA_Team', 0)), 
                    current.get('Ply', current.get('Ply_Team', 0)), 
                    current.get('Y/P', current.get('Y/P_Team', 0)),
                    
                    # Trends (15 features)
                    current.get('Att_Change_1yr', 0), current.get('Yds_Change_1yr', 0),
                    current.get('YpA_Change_1yr', 0), current.get('TD_Change_1yr', 0),
                    current.get('Rec_Change_1yr', 0), current.get('Rec_Yds_Change_1yr', 0),
                    current.get('Rec_TD_Change_1yr', 0), current.get('Total_Yds_Change_1yr', 0),
                    current.get('Total_Touches_Change_1yr', 0), current.get('Total_TD_Change_1yr', 0),
                    current.get('Age_Change', 1), current.get('Career_Progression', 0),
                    current.get('Att_Change_2yr', 0), current.get('Total_Yds_Change_2yr', 0),
                    current.get('YpA_Trend_2yr', 0)
                ]
                
                # Targets (rushing stats only for prediction)
                targets = [
                    int(next_year.get('Att', 0)),
                    int(next_year.get('Yds', 0)),
                    int(next_year.get('TD', 0))
                ]
                
                X_rows.append(features)
                y_rows.append(targets)
    
    # Create DataFrames with comprehensive feature names (62 total features)
    feature_names = [
        # Core rushing (9)
        'Att_N', 'Yds_N', 'TD_N', 'Age_N', 'G_N', 'GS_N', 'YpA_N', 'YpG_N', 'Fmb_N',
        # Core receiving (6)
        'Rec_N', 'Rec_Yds_N', 'Rec_TD_N', 'Rec_YpR_N', 'Rec_Tgt_N', 'Rec_YpTgt_N',
        # Combined metrics (5)
        'Total_Yds_N', 'Total_TD_N', 'Total_Touches_N', 'Total_YpT_N', 'Total_YpG_N',
        # Efficiency (8)
        'Att_per_Game', 'TD_per_Att', 'Fmb_Rate', 'Games_Started_Pct',
        'Rec_per_Game', 'Rec_TD_per_Rec', 'Rec_Yds_per_Game', 'Catch_Rate',
        # Usage/Role (6)
        'High_Usage_Rush', 'Workhorse_Rush', 'High_Usage_Rec', 'Dual_Threat',
        'High_Touch_Total', 'Goal_Line_Back',
        # Versatility (4)
        'Rush_Rec_Ratio', 'Receiving_Dependence', 'Efficiency_Score', 'Volume_Score',
        # Age (4)
        'Age_Squared', 'Prime_Age', 'Rookie', 'Veteran',
        # Success (2)
        'Success_Rate_Adj', 'First_Down_Rate',
        # Team (3)
        'Team_PA', 'Team_Ply', 'Team_YpP',
        # Trends (15)
        'Att_Change_1yr', 'Yds_Change_1yr', 'YpA_Change_1yr', 'TD_Change_1yr',
        'Rec_Change_1yr', 'Rec_Yds_Change_1yr', 'Rec_TD_Change_1yr', 'Total_Yds_Change_1yr',
        'Total_Touches_Change_1yr', 'Total_TD_Change_1yr', 'Age_Change', 'Career_Progression',
        'Att_Change_2yr', 'Total_Yds_Change_2yr', 'YpA_Trend_2yr'
    ]
    
    X = pd.DataFrame(X_rows, columns=feature_names)
    y = pd.DataFrame(y_rows, columns=['Att_N1', 'Yds_N1', 'TD_N1'])
    
    print(f"  ✅ Created {len(X)} training samples with {len(feature_names)} comprehensive features")
    
    return X, y, enhanced_df

def train_models_fast(X, y):
    """Train models quickly with optimized settings."""
    print("🤖 Training models...")
    
    # Quick feature scaling
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Feature selection - need to handle multi-output properly
    k_features = min(30, len(X.columns))
    
    # Use the first target (Att) for feature selection, then apply to all targets
    selector = SelectKBest(f_regression, k=k_features)
    selector.fit(X_scaled, y.iloc[:, 0])  # Use first column (Att) for selection
    X_selected = selector.transform(X_scaled)
    
    # Get selected feature names for debugging
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"  Selected top {len(selected_features)} features:")
    for i, feat in enumerate(selected_features[:15]):
        print(f"    {i+1:2d}. {feat}")
    if len(selected_features) > 15:
        print(f"    ... and {len(selected_features) - 15} more")
    
    # Enhanced model ensemble - use MultiOutputRegressor properly
    models = {
        'ridge': MultiOutputRegressor(Ridge(alpha=3.0)),
        'rf_optimized': MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42
            )
        ),
        'gbm_optimized': MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.08,
                subsample=0.8,
                random_state=42
            )
        )
    }
    
    # Train models
    trained_models = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        try:
            # Convert y to numpy array to ensure compatibility
            y_array = y.values if hasattr(y, 'values') else y
            model.fit(X_selected, y_array)
            trained_models[name] = model
            print(f"    ✅ {name} trained successfully")
        except Exception as e:
            print(f"    ❌ {name} failed: {e}")
            # Try individual regressors if MultiOutputRegressor fails
            try:
                print(f"    🔄 Trying individual regressors for {name}...")
                if 'ridge' in name:
                    individual_models = [Ridge(alpha=3.0) for _ in range(y.shape[1])]
                elif 'rf' in name:
                    individual_models = [RandomForestRegressor(
                        n_estimators=150, max_depth=12, min_samples_split=10,
                        min_samples_leaf=4, max_features='sqrt', n_jobs=-1, random_state=42
                    ) for _ in range(y.shape[1])]
                elif 'gbm' in name:
                    individual_models = [GradientBoostingRegressor(
                        n_estimators=100, max_depth=8, learning_rate=0.08,
                        subsample=0.8, random_state=42
                    ) for _ in range(y.shape[1])]
                
                # Train individual models for each target
                for i, target_model in enumerate(individual_models):
                    target_model.fit(X_selected, y.iloc[:, i] if hasattr(y, 'iloc') else y[:, i])
                
                trained_models[name] = individual_models
                print(f"    ✅ {name} trained with individual regressors")
            except Exception as e2:
                print(f"    ❌ {name} completely failed: {e2}")
    
    return trained_models, scaler, selector

def ensemble_predict_fast(models, X, scaler, selector):
    """Fast ensemble predictions with proper handling of different model types."""
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    X_selected = selector.transform(X_scaled)
    
    predictions = []
    weights = {'ridge': 0.2, 'rf_optimized': 0.4, 'gbm_optimized': 0.4}  # Weighted ensemble
    
    for name, model in models.items():
        weight = weights.get(name, 1.0 / len(models))
        
        try:
            # Check if it's a MultiOutputRegressor or list of individual models
            if hasattr(model, 'predict'):
                # Standard MultiOutputRegressor
                pred = model.predict(X_selected)
            elif isinstance(model, list):
                # List of individual regressors
                pred = np.column_stack([m.predict(X_selected) for m in model])
            else:
                print(f"    ⚠️ Unknown model type for {name}, skipping")
                continue
            
            # Ensure prediction has correct shape
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            
            predictions.append(pred * weight)
        except Exception as e:
            print(f"    ⚠️ Prediction failed for {name}: {e}")
            continue
    
    if not predictions:
        # Fallback: return zeros if all models failed
        print("    ❌ All models failed, returning zero predictions")
        return np.zeros((X_selected.shape[0], 3))
    
    # Weighted average ensemble
    ensemble_pred = np.sum(predictions, axis=0)
    
    return ensemble_pred

def safe_get_stat(row, stat_name, default=0):
    """Safely get a stat from a row with fallbacks."""
    if stat_name in row and not pd.isna(row[stat_name]):
        return row[stat_name]
    return default

def predict_next_season(models, scaler, selector, data_dict, current_year=2024):
    """Predict next season for current top players using comprehensive features."""
    print(f"🔮 Predicting {current_year} season...")
    
    combined_df = data_dict['combined']
    
    # Get most recent year of data
    latest_year = combined_df['Year'].max()
    print(f"  Using {latest_year} data to predict {current_year}")
    
    current_data = combined_df[combined_df['Year'] == latest_year].copy()
    current_enhanced = calculate_comprehensive_features(current_data)
    current_enhanced = current_enhanced.fillna(0)
    
    # Get top players by total touches (rushing + receiving)
    current_enhanced['Total_Usage'] = current_enhanced['Att'] + current_enhanced['Rec_Rec']
    top_players = current_enhanced[current_enhanced['Att'] >= 50].nlargest(30, 'Total_Usage')
    
    predictions_list = []
    
    for _, player in top_players.iterrows():
        # Create comprehensive feature vector using exact column structure
        features = [
            # Core rushing stats (9)
            safe_get_stat(player, 'Att', 0), safe_get_stat(player, 'Yds', 0), 
            safe_get_stat(player, 'TD', 0), safe_get_stat(player, 'Age', 25), 
            safe_get_stat(player, 'G', 0), safe_get_stat(player, 'GS', 0),
            safe_get_stat(player, 'Y/A', 0), safe_get_stat(player, 'Y/G', 0), 
            safe_get_stat(player, 'Fmb', 0),
            
            # Core receiving stats (6)
            safe_get_stat(player, 'Rec_Rec', 0), safe_get_stat(player, 'Rec_Yds', 0), 
            safe_get_stat(player, 'Rec_TD', 0), safe_get_stat(player, 'Rec_Y/R', 0),
            safe_get_stat(player, 'Rec_Tgt', 0), safe_get_stat(player, 'Rec_Y/Tgt', 0),
            
            # Combined metrics (5)
            safe_get_stat(player, 'Total_Yds', 0), safe_get_stat(player, 'Total_TD', 0), 
            safe_get_stat(player, 'Total_Touches', 0), safe_get_stat(player, 'Total_Yds_per_Touch', 0),
            safe_get_stat(player, 'Total_Yds_per_Game', 0),
            
            # Efficiency metrics (8)
            safe_get_stat(player, 'Att_per_Game', 0), safe_get_stat(player, 'TD_per_Att', 0), 
            safe_get_stat(player, 'Fmb_Rate', 0), safe_get_stat(player, 'Games_Started_Pct', 0),
            safe_get_stat(player, 'Rec_per_Game', 0), safe_get_stat(player, 'Rec_TD_per_Rec', 0),
            safe_get_stat(player, 'Rec_Yds_per_Game', 0), safe_get_stat(player, 'Catch_Rate', 0),
            
            # Usage and role indicators (6)
            safe_get_stat(player, 'High_Usage_Rush', 0), safe_get_stat(player, 'Workhorse_Rush', 0),
            safe_get_stat(player, 'High_Usage_Rec', 0), safe_get_stat(player, 'Dual_Threat', 0),
            safe_get_stat(player, 'High_Touch_Total', 0), safe_get_stat(player, 'Goal_Line_Back', 0),
            
            # Versatility metrics (4)
            safe_get_stat(player, 'Rush_Rec_Ratio', 0), safe_get_stat(player, 'Receiving_Dependence', 0),
            safe_get_stat(player, 'Efficiency_Score', 0), safe_get_stat(player, 'Volume_Score', 0),
            
            # Age features (4)
            safe_get_stat(player, 'Age_Squared', 0), safe_get_stat(player, 'Prime_Age', 0), 
            safe_get_stat(player, 'Rookie', 0), safe_get_stat(player, 'Veteran', 0),
            
            # Success metrics (2)
            safe_get_stat(player, 'Success_Rate_Adj', 0), safe_get_stat(player, 'First_Down_Rate', 0),
            
            # Team context (3)
            safe_get_stat(player, 'PA', 0), safe_get_stat(player, 'Ply', 0), 
            safe_get_stat(player, 'Y/P', 0),
            
            # Trends (15) - these will be 0 for latest year predictions
            safe_get_stat(player, 'Att_Change_1yr', 0), safe_get_stat(player, 'Yds_Change_1yr', 0),
            safe_get_stat(player, 'YpA_Change_1yr', 0), safe_get_stat(player, 'TD_Change_1yr', 0),
            safe_get_stat(player, 'Rec_Change_1yr', 0), safe_get_stat(player, 'Rec_Yds_Change_1yr', 0),
            safe_get_stat(player, 'Rec_TD_Change_1yr', 0), safe_get_stat(player, 'Total_Yds_Change_1yr', 0),
            safe_get_stat(player, 'Total_Touches_Change_1yr', 0), safe_get_stat(player, 'Total_TD_Change_1yr', 0),
            safe_get_stat(player, 'Age_Change', 1), safe_get_stat(player, 'Career_Progression', 0),
            safe_get_stat(player, 'Att_Change_2yr', 0), safe_get_stat(player, 'Total_Yds_Change_2yr', 0),
            safe_get_stat(player, 'YpA_Trend_2yr', 0)
        ]
        
        # Create DataFrame for prediction (must match training feature names exactly)
        feature_names = [
            # Core rushing (9)
            'Att_N', 'Yds_N', 'TD_N', 'Age_N', 'G_N', 'GS_N', 'YpA_N', 'YpG_N', 'Fmb_N',
            # Core receiving (6)
            'Rec_N', 'Rec_Yds_N', 'Rec_TD_N', 'Rec_YpR_N', 'Rec_Tgt_N', 'Rec_YpTgt_N',
            # Combined metrics (5)
            'Total_Yds_N', 'Total_TD_N', 'Total_Touches_N', 'Total_YpT_N', 'Total_YpG_N',
            # Efficiency (8)
            'Att_per_Game', 'TD_per_Att', 'Fmb_Rate', 'Games_Started_Pct',
            'Rec_per_Game', 'Rec_TD_per_Rec', 'Rec_Yds_per_Game', 'Catch_Rate',
            # Usage/Role (6)
            'High_Usage_Rush', 'Workhorse_Rush', 'High_Usage_Rec', 'Dual_Threat',
            'High_Touch_Total', 'Goal_Line_Back',
            # Versatility (4)
            'Rush_Rec_Ratio', 'Receiving_Dependence', 'Efficiency_Score', 'Volume_Score',
            # Age (4)
            'Age_Squared', 'Prime_Age', 'Rookie', 'Veteran',
            # Success (2)
            'Success_Rate_Adj', 'First_Down_Rate',
            # Team (3)
            'Team_PA', 'Team_Ply', 'Team_YpP',
            # Trends (15)
            'Att_Change_1yr', 'Yds_Change_1yr', 'YpA_Change_1yr', 'TD_Change_1yr',
            'Rec_Change_1yr', 'Rec_Yds_Change_1yr', 'Rec_TD_Change_1yr', 'Total_Yds_Change_1yr',
            'Total_Touches_Change_1yr', 'Total_TD_Change_1yr', 'Age_Change', 'Career_Progression',
            'Att_Change_2yr', 'Total_Yds_Change_2yr', 'YpA_Trend_2yr'
        ]
        
        X_pred = pd.DataFrame([features], columns=feature_names)
        
        # Make prediction
        pred = ensemble_predict_fast(models, X_pred, scaler, selector)[0]
        
        prediction = {
            'Player': safe_get_stat(player, 'Player', 'Unknown'),
            'Team': safe_get_stat(player, 'Tm', 'UNK'),
            'Position': safe_get_stat(player, 'Pos', 'RB'),
            'Age_Current': int(safe_get_stat(player, 'Age', 25)),
            'Age_Next': int(safe_get_stat(player, 'Age', 25) + 1),
            f'{latest_year}_Att': int(safe_get_stat(player, 'Att', 0)),
            f'{latest_year}_Yds': int(safe_get_stat(player, 'Yds', 0)),
            f'{latest_year}_TD': int(safe_get_stat(player, 'TD', 0)),
            f'{latest_year}_YpA': round(safe_get_stat(player, 'Y/A', 0), 1),
            f'{latest_year}_Rec': int(safe_get_stat(player, 'Rec_Rec', 0)),
            f'{latest_year}_Rec_Yds': int(safe_get_stat(player, 'Rec_Yds', 0)),
            f'{latest_year}_Rec_TD': int(safe_get_stat(player, 'Rec_TD', 0)),
            f'{latest_year}_Total_Yds': int(safe_get_stat(player, 'Total_Yds', 0)),
            f'{latest_year}_Total_Touches': int(safe_get_stat(player, 'Total_Touches', 0)),
            f'{current_year}_Pred_Att': max(0, int(pred[0])),
            f'{current_year}_Pred_Yds': max(0, int(pred[1])),
            f'{current_year}_Pred_TD': max(0, int(pred[2])),
        }
        
        # Calculate predicted Y/A
        if prediction[f'{current_year}_Pred_Att'] > 0:
            prediction[f'{current_year}_Pred_YpA'] = round(prediction[f'{current_year}_Pred_Yds'] / prediction[f'{current_year}_Pred_Att'], 1)
        else:
            prediction[f'{current_year}_Pred_YpA'] = 0.0
        
        predictions_list.append(prediction)
    
    # Create results DataFrame
    predictions_df = pd.DataFrame(predictions_list)
    
    # Sort by predicted yards
    predictions_df = predictions_df.sort_values(f'{current_year}_Pred_Yds', ascending=False)
    
    print(f"  ✅ Generated predictions for {len(predictions_df)} players")
    
    return predictions_df

def quick_backtest(models, scaler, selector, data_dict, test_years=[2022, 2023]):
    """Quick backtest on recent years using comprehensive features."""
    print(f"🔍 Quick backtest on years: {test_years}")
    
    combined_df = data_dict['combined']
    results = []
    
    for test_year in test_years:
        print(f"  Testing {test_year - 1} → {test_year}")
        
        # Get data for prediction year and target year
        pred_data = combined_df[combined_df['Year'] == test_year - 1].copy()
        target_data = combined_df[combined_df['Year'] == test_year].copy()
        
        if pred_data.empty or target_data.empty:
            continue
        
        # Add features
        pred_enhanced = calculate_comprehensive_features(pred_data)
        pred_enhanced = pred_enhanced.fillna(0)
        
        # Get top players by total usage
        pred_enhanced['Total_Usage'] = pred_enhanced['Att'] + pred_enhanced['Rec_Rec']
        top_players = pred_enhanced[pred_enhanced['Att'] >= 50].nlargest(25, 'Total_Usage')
        
        for _, player in top_players.iterrows():
            player_name = safe_get_stat(player, 'Player', 'Unknown')
            
            # Create feature vector (same structure as prediction)
            features = [
                # Core rushing stats (9)
                safe_get_stat(player, 'Att', 0), safe_get_stat(player, 'Yds', 0), 
                safe_get_stat(player, 'TD', 0), safe_get_stat(player, 'Age', 25), 
                safe_get_stat(player, 'G', 0), safe_get_stat(player, 'GS', 0),
                safe_get_stat(player, 'Y/A', 0), safe_get_stat(player, 'Y/G', 0), 
                safe_get_stat(player, 'Fmb', 0),
                
                # Core receiving stats (6)
                safe_get_stat(player, 'Rec_Rec', 0), safe_get_stat(player, 'Rec_Yds', 0), 
                safe_get_stat(player, 'Rec_TD', 0), safe_get_stat(player, 'Rec_Y/R', 0),
                safe_get_stat(player, 'Rec_Tgt', 0), safe_get_stat(player, 'Rec_Y/Tgt', 0),
                
                # Combined metrics (5)
                safe_get_stat(player, 'Total_Yds', 0), safe_get_stat(player, 'Total_TD', 0), 
                safe_get_stat(player, 'Total_Touches', 0), safe_get_stat(player, 'Total_Yds_per_Touch', 0),
                safe_get_stat(player, 'Total_Yds_per_Game', 0),
                
                # Efficiency metrics (8)
                safe_get_stat(player, 'Att_per_Game', 0), safe_get_stat(player, 'TD_per_Att', 0), 
                safe_get_stat(player, 'Fmb_Rate', 0), safe_get_stat(player, 'Games_Started_Pct', 0),
                safe_get_stat(player, 'Rec_per_Game', 0), safe_get_stat(player, 'Rec_TD_per_Rec', 0),
                safe_get_stat(player, 'Rec_Yds_per_Game', 0), safe_get_stat(player, 'Catch_Rate', 0),
                
                # Usage and role indicators (6)
                safe_get_stat(player, 'High_Usage_Rush', 0), safe_get_stat(player, 'Workhorse_Rush', 0),
                safe_get_stat(player, 'High_Usage_Rec', 0), safe_get_stat(player, 'Dual_Threat', 0),
                safe_get_stat(player, 'High_Touch_Total', 0), safe_get_stat(player, 'Goal_Line_Back', 0),
                
                # Versatility metrics (4)
                safe_get_stat(player, 'Rush_Rec_Ratio', 0), safe_get_stat(player, 'Receiving_Dependence', 0),
                safe_get_stat(player, 'Efficiency_Score', 0), safe_get_stat(player, 'Volume_Score', 0),
                
                # Age features (4)
                safe_get_stat(player, 'Age_Squared', 0), safe_get_stat(player, 'Prime_Age', 0), 
                safe_get_stat(player, 'Rookie', 0), safe_get_stat(player, 'Veteran', 0),
                
                # Success metrics (2)
                safe_get_stat(player, 'Success_Rate_Adj', 0), safe_get_stat(player, 'First_Down_Rate', 0),
                
                # Team context (3)
                0, 0, 0,  # Team placeholders for backtest
                
                # Trends (15)
                safe_get_stat(player, 'Att_Change_1yr', 0), safe_get_stat(player, 'Yds_Change_1yr', 0),
                safe_get_stat(player, 'YpA_Change_1yr', 0), safe_get_stat(player, 'TD_Change_1yr', 0),
                safe_get_stat(player, 'Rec_Change_1yr', 0), safe_get_stat(player, 'Rec_Yds_Change_1yr', 0),
                safe_get_stat(player, 'Rec_TD_Change_1yr', 0), safe_get_stat(player, 'Total_Yds_Change_1yr', 0),
                safe_get_stat(player, 'Total_Touches_Change_1yr', 0), safe_get_stat(player, 'Total_TD_Change_1yr', 0),
                safe_get_stat(player, 'Age_Change', 1), safe_get_stat(player, 'Career_Progression', 0),
                safe_get_stat(player, 'Att_Change_2yr', 0), safe_get_stat(player, 'Total_Yds_Change_2yr', 0),
                safe_get_stat(player, 'YpA_Trend_2yr', 0)
            ]
            
            # Create feature DataFrame
            feature_names = [
                # Core rushing (9)
                'Att_N', 'Yds_N', 'TD_N', 'Age_N', 'G_N', 'GS_N', 'YpA_N', 'YpG_N', 'Fmb_N',
                # Core receiving (6)
                'Rec_N', 'Rec_Yds_N', 'Rec_TD_N', 'Rec_YpR_N', 'Rec_Tgt_N', 'Rec_YpTgt_N',
                # Combined metrics (5)
                'Total_Yds_N', 'Total_TD_N', 'Total_Touches_N', 'Total_YpT_N', 'Total_YpG_N',
                # Efficiency (8)
                'Att_per_Game', 'TD_per_Att', 'Fmb_Rate', 'Games_Started_Pct',
                'Rec_per_Game', 'Rec_TD_per_Rec', 'Rec_Yds_per_Game', 'Catch_Rate',
                # Usage/Role (6)
                'High_Usage_Rush', 'Workhorse_Rush', 'High_Usage_Rec', 'Dual_Threat',
                'High_Touch_Total', 'Goal_Line_Back',
                # Versatility (4)
                'Rush_Rec_Ratio', 'Receiving_Dependence', 'Efficiency_Score', 'Volume_Score',
                # Age (4)
                'Age_Squared', 'Prime_Age', 'Rookie', 'Veteran',
                # Success (2)
                'Success_Rate_Adj', 'First_Down_Rate',
                # Team (3)
                'Team_PA', 'Team_Ply', 'Team_YpP',
                # Trends (15)
                'Att_Change_1yr', 'Yds_Change_1yr', 'YpA_Change_1yr', 'TD_Change_1yr',
                'Rec_Change_1yr', 'Rec_Yds_Change_1yr', 'Rec_TD_Change_1yr', 'Total_Yds_Change_1yr',
                'Total_Touches_Change_1yr', 'Total_TD_Change_1yr', 'Age_Change', 'Career_Progression',
                'Att_Change_2yr', 'Total_Yds_Change_2yr', 'YpA_Trend_2yr'
            ]
            
            X_test_df = pd.DataFrame([features], columns=feature_names)
            
            # Make prediction
            pred = ensemble_predict_fast(models, X_test_df, scaler, selector)[0]
            
            # Find actual results
            actual = target_data[target_data['Player'] == player_name]
            if not actual.empty:
                actual_stats = actual.iloc[0]
                
                result = {
                    'Year': test_year,
                    'Player': player_name,
                    'Age': int(safe_get_stat(player, 'Age', 25) + 1),
                    'Pred_Att': int(pred[0]),
                    'Actual_Att': int(safe_get_stat(actual_stats, 'Att', 0)),
                    'Pred_Yds': int(pred[1]),
                    'Actual_Yds': int(safe_get_stat(actual_stats, 'Yds', 0)),
                    'Pred_TD': int(pred[2]),
                    'Actual_TD': int(safe_get_stat(actual_stats, 'TD', 0))
                }
                
                # Calculate errors
                result['Att_Error'] = abs(result['Pred_Att'] - result['Actual_Att'])
                result['Yds_Error'] = abs(result['Pred_Yds'] - result['Actual_Yds'])
                result['TD_Error'] = abs(result['Pred_TD'] - result['Actual_TD'])
                
                # Calculate percentage errors
                if result['Actual_Att'] > 0:
                    result['Att_Error_Pct'] = round(100 * result['Att_Error'] / result['Actual_Att'], 1)
                else:
                    result['Att_Error_Pct'] = 0
                    
                if result['Actual_Yds'] > 0:
                    result['Yds_Error_Pct'] = round(100 * result['Yds_Error'] / result['Actual_Yds'], 1)
                else:
                    result['Yds_Error_Pct'] = 0
                
                results.append(result)
    
    if results:
        results_df = pd.DataFrame(results)
        
        # Calculate average errors
        avg_att_error = results_df['Att_Error'].mean()
        avg_yds_error = results_df['Yds_Error'].mean()
        avg_td_error = results_df['TD_Error'].mean()
        avg_att_error_pct = results_df['Att_Error_Pct'].mean()
        avg_yds_error_pct = results_df['Yds_Error_Pct'].mean()
        
        print(f"  📊 Backtest Results:")
        print(f"    Average Attempts Error: {avg_att_error:.1f} ({avg_att_error_pct:.1f}%)")
        print(f"    Average Yards Error: {avg_yds_error:.1f} ({avg_yds_error_pct:.1f}%)")
        print(f"    Average TD Error: {avg_td_error:.1f}")
        print(f"    Sample size: {len(results_df)} predictions")
        
        return results_df
    
    return pd.DataFrame()

def display_comprehensive_predictions(predictions_df, current_year=2024):
    """Display comprehensive predictions including receiving data."""
    if predictions_df.empty:
        print("No predictions to display.")
        return
        
    # Get the actual column names from the DataFrame
    cols = list(predictions_df.columns)
    latest_year = current_year - 1
    
    print(f"\n🏈 TOP {current_year} NFL RUSHING PREDICTIONS")
    print("=" * 130)
    print(f"Debug - Available columns: {cols}")  # Debug line
    
    # Find the actual column names for the latest year stats
    att_col = f'{latest_year}_Att'
    yds_col = f'{latest_year}_Yds'
    td_col = f'{latest_year}_TD'
    rec_col = f'{latest_year}_Rec'
    rec_yds_col = f'{latest_year}_Rec_Yds'
    total_yds_col = f'{latest_year}_Total_Yds'
    
    # Find prediction columns
    pred_att_col = f'{current_year}_Pred_Att'
    pred_yds_col = f'{current_year}_Pred_Yds'
    pred_td_col = f'{current_year}_Pred_TD'
    pred_ypa_col = f'{current_year}_Pred_YpA'
    
    # Check if columns exist
    missing_cols = []
    for col_name, col_var in [
        (att_col, 'att'), (yds_col, 'yds'), (td_col, 'td'),
        (pred_att_col, 'pred_att'), (pred_yds_col, 'pred_yds'), (pred_td_col, 'pred_td')
    ]:
        if col_name not in cols:
            missing_cols.append(col_name)
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"Available columns: {cols}")
        return
    
    print(f"{'Rank':<4} {'Player':<20} {'Pos':<3} {'Team':<4} {'Age':<3} "
          f"{latest_year} Performance              {current_year} Predictions")
    print(f"{'':4} {'':20} {'':3} {'':4} {'':3} "
          f"{'Rush':<12} {'Rec':<8} {'Total':<6} {'Rush':<14} {'Y/A':<4}")
    print(f"{'':4} {'':20} {'':3} {'':4} {'':3} "
          f"{'Att/Yds/TD':<12} {'Rec/Yds':<8} {'Yds':<6} {'Att/Yds/TD':<14} {'':4}")
    print("-" * 130)
    
    # Display each row with better error handling
    for i, (idx, row) in enumerate(predictions_df.head(25).iterrows()):
        try:
            # Get basic info with safe fallbacks
            player_name = str(row.get('Player', 'Unknown'))[:19]
            pos = str(row.get('Position', row.get('Pos', 'RB')))[:3]
            team = str(row.get('Team', 'UNK'))[:4]
            age = int(row.get('Age_Next', row.get('Age_Current', 25)))
            
            # Get rush stats safely
            att_curr = int(row.get(att_col, 0))
            yds_curr = int(row.get(yds_col, 0))
            td_curr = int(row.get(td_col, 0))
            rush_current = f"{att_curr}/{yds_curr}/{td_curr}"
            
            # Get receiving stats safely
            rec_curr = int(row.get(rec_col, 0))
            rec_yds_curr = int(row.get(rec_yds_col, 0))
            rec_current = f"{rec_curr}/{rec_yds_curr}"
            
            # Get total yards safely
            total_yds = int(row.get(total_yds_col, yds_curr + rec_yds_curr))
            
            # Get predictions safely
            pred_att = int(row.get(pred_att_col, 0))
            pred_yds = int(row.get(pred_yds_col, 0))
            pred_td = int(row.get(pred_td_col, 0))
            rush_pred = f"{pred_att}/{pred_yds}/{pred_td}"
            
            # Get Y/A safely
            pred_ypa = row.get(pred_ypa_col, 0.0)
            if isinstance(pred_ypa, (int, float)):
                ypa_str = f"{pred_ypa:.1f}"
            else:
                ypa_str = "0.0"
            
            print(f"{i+1:<4} {player_name:<20} {pos:<3} {team:<4} {age:<3} "
                  f"{rush_current:<12} {rec_current:<8} {total_yds:<6} "
                  f"{rush_pred:<14} {ypa_str}")
                  
        except Exception as e:
            print(f"Error displaying row {i}: {e}")
            print(f"Row data: {dict(row)}")
            break  # Stop on first error to debug
    
    print(f"\nDisplayed {min(25, len(predictions_df))} predictions")

def main():
    """Main execution function."""
    print("🏈 NFL Comprehensive Rushing Predictor v2.0")
    print("=" * 60)
    print("Optimized for your exact data structure")
    print()
    
    # Load and merge data
    data = load_and_merge_data()
    if data is None:
        return
    
    # Build comprehensive training data
    X, y, enhanced_df = build_comprehensive_training_data(data, min_year=2000, max_year=2023)
    
    if len(X) == 0:
        print("❌ No training data created. Check your data files.")
        return
    
    # Train models
    models, scaler, selector = train_models_fast(X, y)
    
    if not models:
        print("❌ No models trained successfully.")
        return
    
    # Quick backtest
    backtest_df = quick_backtest(models, scaler, selector, data, test_years=[2022, 2023])
    
    # Generate predictions for next season
    predictions_df = predict_next_season(models, scaler, selector, data, current_year=2024)
    
    # Debug: Print column names
    if not predictions_df.empty:
        print(f"\n🔍 Predictions DataFrame columns: {list(predictions_df.columns)}")
        print(f"DataFrame shape: {predictions_df.shape}")
        print(f"Sample row:\n{predictions_df.iloc[0] if len(predictions_df) > 0 else 'No data'}")
    
    # Display comprehensive results
    display_comprehensive_predictions(predictions_df, current_year=2024)
    
    # Save predictions to CSV
    if not predictions_df.empty:
        output_file = DATA_DIR / "comprehensive_rushing_predictions_2024.csv"
        predictions_df.to_csv(output_file, index=False)
        print(f"\n💾 Comprehensive predictions saved to {output_file}")
    
    if not backtest_df.empty:
        backtest_file = DATA_DIR / "comprehensive_backtest_results.csv"
        backtest_df.to_csv(backtest_file, index=False)
        print(f"💾 Backtest results saved to {backtest_file}")
        
        # Show some backtest examples
        print(f"\n📊 Sample Backtest Results:")
        print("-" * 80)
        print(f"{'Player':<18} {'Year':<5} {'Predicted':<15} {'Actual':<15} {'Error':<10}")
        print(f"{'':18} {'':5} {'Att/Yds/TD':<15} {'Att/Yds/TD':<15} {'Yds':<10}")
        print("-" * 80)
        
        for _, row in backtest_df.head(10).iterrows():
            pred_str = f"{row['Pred_Att']}/{row['Pred_Yds']}/{row['Pred_TD']}"
            actual_str = f"{row['Actual_Att']}/{row['Actual_Yds']}/{row['Actual_TD']}"
            print(f"{row['Player'][:17]:<18} {row['Year']:<5} {pred_str:<15} {actual_str:<15} {row['Yds_Error']:<10}")
    else:
        print("\n⚠️ No backtest results generated")

if __name__ == "__main__":
    main()