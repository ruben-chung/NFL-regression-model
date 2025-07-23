"""
enhanced_rushing_predictor.py

Comprehensive ML analysis with extensive backtesting (1990-2010 train, 2010-2024 test)
Includes detailed performance metrics: R², MAE, RMSE, correlation analysis, and more.
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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
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

def build_comprehensive_training_data(data_dict, min_year=1990, max_year=2010):
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

def train_models_comprehensive(X, y):
    """Train models with comprehensive settings for backtesting."""
    print("🤖 Training comprehensive models...")
    
    # Feature scaling
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Feature selection
    k_features = min(35, len(X.columns))
    
    # Use the first target (Att) for feature selection, then apply to all targets
    selector = SelectKBest(f_regression, k=k_features)
    selector.fit(X_scaled, y.iloc[:, 0])
    X_selected = selector.transform(X_scaled)
    
    # Get selected feature names for analysis
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"  Selected top {len(selected_features)} features:")
    for i, feat in enumerate(selected_features[:20]):
        print(f"    {i+1:2d}. {feat}")
    if len(selected_features) > 20:
        print(f"    ... and {len(selected_features) - 20} more")
    
    # Enhanced model ensemble
    models = {
        'ridge': MultiOutputRegressor(Ridge(alpha=2.0)),
        'lasso': MultiOutputRegressor(Lasso(alpha=0.5)),
        'rf_optimized': MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=8,
                min_samples_leaf=3,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42
            )
        ),
        'gbm_optimized': MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=150,
                max_depth=10,
                learning_rate=0.07,
                subsample=0.85,
                random_state=42
            )
        ),
        'mlp': MultiOutputRegressor(
            MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        )
    }
    
    # Train models with error handling
    trained_models = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        try:
            y_array = y.values if hasattr(y, 'values') else y
            model.fit(X_selected, y_array)
            trained_models[name] = model
            print(f"    ✅ {name} trained successfully")
        except Exception as e:
            print(f"    ❌ {name} failed: {e}")
    
    return trained_models, scaler, selector, selected_features

def ensemble_predict_comprehensive(models, X, scaler, selector):
    """Comprehensive ensemble predictions with weighted voting."""
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    X_selected = selector.transform(X_scaled)
    
    predictions = []
    weights = {
        'ridge': 0.15, 
        'lasso': 0.1,
        'rf_optimized': 0.35, 
        'gbm_optimized': 0.35,
        'mlp': 0.05
    }
    
    for name, model in models.items():
        weight = weights.get(name, 1.0 / len(models))
        
        try:
            pred = model.predict(X_selected)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            predictions.append(pred * weight)
        except Exception as e:
            print(f"    ⚠️ Prediction failed for {name}: {e}")
            continue
    
    if not predictions:
        return np.zeros((X_selected.shape[0], 3))
    
    ensemble_pred = np.sum(predictions, axis=0)
    return ensemble_pred

def comprehensive_backtest(data_dict, train_years=(1990, 2010), test_years=(2010, 2024)):
    """Comprehensive backtesting with detailed metrics."""
    print(f"🔍 COMPREHENSIVE BACKTESTING")
    print(f"Training Period: {train_years[0]}-{train_years[1]}")
    print(f"Testing Period: {test_years[0]}-{test_years[1]}")
    print("=" * 80)
    
    # Build training data
    X_train, y_train, _ = build_comprehensive_training_data(
        data_dict, min_year=train_years[0], max_year=train_years[1]
    )
    
    if len(X_train) == 0:
        print("❌ No training data available")
        return None, None
    
    # Train models
    models, scaler, selector, selected_features = train_models_comprehensive(X_train, y_train)
    
    if not models:
        print("❌ No models trained successfully")
        return None, None
    
    # Now test on each year in the test period
    combined_df = data_dict['combined']
    all_predictions = []
    all_actuals = []
    detailed_results = []
    
    for test_year in range(test_years[0] + 1, test_years[1] + 1):
        print(f"\n  Testing {test_year - 1} → {test_year}")
        
        # Get data for prediction year and target year
        pred_data = combined_df[combined_df['Year'] == test_year - 1].copy()
        target_data = combined_df[combined_df['Year'] == test_year].copy()
        
        if pred_data.empty or target_data.empty:
            print(f"    ⚠️ No data available for {test_year}")
            continue
        
        # Add features to prediction data
        pred_enhanced = calculate_comprehensive_features(pred_data)
        pred_enhanced = pred_enhanced.fillna(0)
        
        # Get players with sufficient usage
        pred_enhanced['Total_Usage'] = pred_enhanced['Att'] + pred_enhanced['Rec_Rec']
        candidates = pred_enhanced[pred_enhanced['Att'] >= 50]
        
        year_predictions = []
        year_actuals = []
        
        for _, player in candidates.iterrows():
            player_name = player.get('Player', 'Unknown')
            
            # Create feature vector
            features = [
                # Core rushing stats (9)
                player.get('Att', 0), player.get('Yds', 0), player.get('TD', 0),
                player.get('Age', 25), player.get('G', 0), player.get('GS', 0),
                player.get('Y/A', 0), player.get('Y/G', 0), player.get('Fmb', 0),
                
                # Core receiving stats (6)
                player.get('Rec_Rec', 0), player.get('Rec_Yds', 0), player.get('Rec_TD', 0),
                player.get('Rec_Y/R', 0), player.get('Rec_Tgt', 0), player.get('Rec_Y/Tgt', 0),
                
                # Combined metrics (5)
                player.get('Total_Yds', 0), player.get('Total_TD', 0), 
                player.get('Total_Touches', 0), player.get('Total_Yds_per_Touch', 0),
                player.get('Total_Yds_per_Game', 0),
                
                # Efficiency metrics (8)
                player.get('Att_per_Game', 0), player.get('TD_per_Att', 0), 
                player.get('Fmb_Rate', 0), player.get('Games_Started_Pct', 0),
                player.get('Rec_per_Game', 0), player.get('Rec_TD_per_Rec', 0),
                player.get('Rec_Yds_per_Game', 0), player.get('Catch_Rate', 0),
                
                # Usage and role indicators (6)
                player.get('High_Usage_Rush', 0), player.get('Workhorse_Rush', 0),
                player.get('High_Usage_Rec', 0), player.get('Dual_Threat', 0),
                player.get('High_Touch_Total', 0), player.get('Goal_Line_Back', 0),
                
                # Versatility metrics (4)
                player.get('Rush_Rec_Ratio', 0), player.get('Receiving_Dependence', 0),
                player.get('Efficiency_Score', 0), player.get('Volume_Score', 0),
                
                # Age features (4)
                player.get('Age_Squared', 0), player.get('Prime_Age', 0), 
                player.get('Rookie', 0), player.get('Veteran', 0),
                
                # Success metrics (2)
                player.get('Success_Rate_Adj', 0), player.get('First_Down_Rate', 0),
                
                # Team context (3)
                0, 0, 0,  # Team placeholders for backtest
                
                # Trends (15)
                player.get('Att_Change_1yr', 0), player.get('Yds_Change_1yr', 0),
                player.get('YpA_Change_1yr', 0), player.get('TD_Change_1yr', 0),
                player.get('Rec_Change_1yr', 0), player.get('Rec_Yds_Change_1yr', 0),
                player.get('Rec_TD_Change_1yr', 0), player.get('Total_Yds_Change_1yr', 0),
                player.get('Total_Touches_Change_1yr', 0), player.get('Total_TD_Change_1yr', 0),
                player.get('Age_Change', 1), player.get('Career_Progression', 0),
                player.get('Att_Change_2yr', 0), player.get('Total_Yds_Change_2yr', 0),
                player.get('YpA_Trend_2yr', 0)
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
            pred = ensemble_predict_comprehensive(models, X_test_df, scaler, selector)[0]
            
            # Find actual results
            actual = target_data[target_data['Player'] == player_name]
            if not actual.empty:
                actual_stats = actual.iloc[0]
                
                # Store predictions and actuals
                pred_values = [max(0, int(pred[0])), max(0, int(pred[1])), max(0, int(pred[2]))]
                actual_values = [
                    int(actual_stats.get('Att', 0)),
                    int(actual_stats.get('Yds', 0)),
                    int(actual_stats.get('TD', 0))
                ]
                
                year_predictions.append(pred_values)
                year_actuals.append(actual_values)
                
                # Store detailed results
                detailed_result = {
                    'Year': test_year,
                    'Player': player_name,
                    'Age': int(player.get('Age', 25) + 1),
                    'Team': player.get('Tm', 'UNK'),
                    'Pred_Att': pred_values[0],
                    'Actual_Att': actual_values[0],
                    'Pred_Yds': pred_values[1],
                    'Actual_Yds': actual_values[1],
                    'Pred_TD': pred_values[2],
                    'Actual_TD': actual_values[2],
                    'Att_Error': abs(pred_values[0] - actual_values[0]),
                    'Yds_Error': abs(pred_values[1] - actual_values[1]),
                    'TD_Error': abs(pred_values[2] - actual_values[2])
                }
                
                # Calculate percentage errors
                if actual_values[0] > 0:
                    detailed_result['Att_Error_Pct'] = 100 * detailed_result['Att_Error'] / actual_values[0]
                else:
                    detailed_result['Att_Error_Pct'] = 0
                    
                if actual_values[1] > 0:
                    detailed_result['Yds_Error_Pct'] = 100 * detailed_result['Yds_Error'] / actual_values[1]
                else:
                    detailed_result['Yds_Error_Pct'] = 0
                
                detailed_results.append(detailed_result)
        
        if year_predictions:
            all_predictions.extend(year_predictions)
            all_actuals.extend(year_actuals)
            print(f"    ✅ {len(year_predictions)} predictions made for {test_year}")
    
    # Convert to numpy arrays for metric calculations
    if all_predictions and all_actuals:
        predictions_array = np.array(all_predictions)
        actuals_array = np.array(all_actuals)
        
        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(predictions_array, actuals_array)
        
        # Create detailed results DataFrame
        results_df = pd.DataFrame(detailed_results)
        
        return metrics, results_df
    else:
        print("❌ No valid predictions made during backtesting")
        return None, None

def calculate_comprehensive_metrics(predictions, actuals):
    """Calculate comprehensive performance metrics."""
    print("\n📊 CALCULATING COMPREHENSIVE METRICS")
    print("=" * 60)
    
    metrics = {}
    target_names = ['Attempts', 'Yards', 'Touchdowns']
    
    for i, target in enumerate(target_names):
        pred_col = predictions[:, i]
        actual_col = actuals[:, i]
        
        # Basic metrics
        mae = mean_absolute_error(actual_col, pred_col)
        mse = mean_squared_error(actual_col, pred_col)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_col, pred_col)
        
        # Correlation metrics
        pearson_corr, pearson_p = pearsonr(pred_col, actual_col)
        spearman_corr, spearman_p = spearmanr(pred_col, actual_col)
        
        # Percentage-based metrics
        mape = np.mean(np.abs((actual_col - pred_col) / np.maximum(actual_col, 1))) * 100
        
        # Accuracy within ranges
        within_10_pct = np.mean(np.abs(pred_col - actual_col) / np.maximum(actual_col, 1) <= 0.10) * 100
        within_20_pct = np.mean(np.abs(pred_col - actual_col) / np.maximum(actual_col, 1) <= 0.20) * 100
        within_30_pct = np.mean(np.abs(pred_col - actual_col) / np.maximum(actual_col, 1) <= 0.30) * 100
        
        # Direction accuracy (for non-zero values)
        if i > 0:  # For yards and TDs, check directional accuracy
            non_zero_mask = actual_col > 0
            if np.sum(non_zero_mask) > 0:
                direction_accuracy = np.mean(
                    np.sign(pred_col[non_zero_mask]) == np.sign(actual_col[non_zero_mask])
                ) * 100
            else:
                direction_accuracy = 0
        else:
            direction_accuracy = 0
        
        # Bias metrics
        bias = np.mean(pred_col - actual_col)
        relative_bias = bias / np.mean(actual_col) * 100 if np.mean(actual_col) > 0 else 0
        
        metrics[target] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'Pearson_Correlation': pearson_corr,
            'Pearson_P_Value': pearson_p,
            'Spearman_Correlation': spearman_corr,
            'Spearman_P_Value': spearman_p,
            'MAPE': mape,
            'Within_10%': within_10_pct,
            'Within_20%': within_20_pct,
            'Within_30%': within_30_pct,
            'Direction_Accuracy': direction_accuracy,
            'Bias': bias,
            'Relative_Bias_%': relative_bias,
            'Mean_Actual': np.mean(actual_col),
            'Mean_Predicted': np.mean(pred_col),
            'Std_Actual': np.std(actual_col),
            'Std_Predicted': np.std(pred_col)
        }
        
        print(f"\n{target.upper()} METRICS:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.1f}%")
        print(f"  Pearson Correlation: {pearson_corr:.4f} (p={pearson_p:.4f})")
        print(f"  Spearman Correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")
        print(f"  Predictions within 20%: {within_20_pct:.1f}%")
        print(f"  Bias: {bias:.2f} ({relative_bias:.1f}%)")
    
    # Overall metrics
    overall_r2 = np.mean([metrics[target]['R²'] for target in target_names])
    overall_mae = np.mean([metrics[target]['MAE'] for target in target_names])
    overall_corr = np.mean([metrics[target]['Pearson_Correlation'] for target in target_names])
    
    metrics['Overall'] = {
        'Average_R²': overall_r2,
        'Average_MAE': overall_mae,
        'Average_Correlation': overall_corr,
        'Total_Predictions': len(predictions)
    }
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Average R²: {overall_r2:.4f}")
    print(f"  Average MAE: {overall_mae:.2f}")
    print(f"  Average Correlation: {overall_corr:.4f}")
    print(f"  Total Predictions: {len(predictions)}")
    
    return metrics

def create_performance_visualizations(metrics, results_df, save_plots=True):
    """Create comprehensive performance visualizations."""
    print("\n📈 CREATING PERFORMANCE VISUALIZATIONS")
    print("=" * 50)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. R² Scores by Target
    ax1 = plt.subplot(3, 4, 1)
    targets = ['Attempts', 'Yards', 'Touchdowns']
    r2_scores = [metrics[target]['R²'] for target in targets]
    bars = ax1.bar(targets, r2_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('R² Scores by Target', fontsize=14, fontweight='bold')
    ax1.set_ylabel('R² Score')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(r2_scores):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. MAE by Target
    ax2 = plt.subplot(3, 4, 2)
    mae_scores = [metrics[target]['MAE'] for target in targets]
    bars = ax2.bar(targets, mae_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_title('Mean Absolute Error by Target', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MAE')
    for i, v in enumerate(mae_scores):
        ax2.text(i, v + max(mae_scores) * 0.02, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Correlation Analysis
    ax3 = plt.subplot(3, 4, 3)
    pearson_corrs = [metrics[target]['Pearson_Correlation'] for target in targets]
    spearman_corrs = [metrics[target]['Spearman_Correlation'] for target in targets]
    x = np.arange(len(targets))
    width = 0.35
    ax3.bar(x - width/2, pearson_corrs, width, label='Pearson', color='skyblue')
    ax3.bar(x + width/2, spearman_corrs, width, label='Spearman', color='lightcoral')
    ax3.set_title('Correlation Analysis', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Correlation Coefficient')
    ax3.set_xticks(x)
    ax3.set_xticklabels(targets)
    ax3.legend()
    ax3.set_ylim(0, 1)
    
    # 4. Accuracy Within Percentage Ranges
    ax4 = plt.subplot(3, 4, 4)
    within_ranges = ['Within_10%', 'Within_20%', 'Within_30%']
    range_labels = ['10%', '20%', '30%']
    yards_accuracy = [metrics['Yards'][range_key] for range_key in within_ranges]
    ax4.plot(range_labels, yards_accuracy, marker='o', linewidth=3, markersize=8, color='orange')
    ax4.set_title('Yards Prediction Accuracy', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Predictions Within Range (%)')
    ax4.set_xlabel('Tolerance Range')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    # 5-7. Scatter plots for each target
    for i, target in enumerate(targets):
        ax = plt.subplot(3, 4, 5 + i)
        
        if target == 'Attempts':
            pred_col = results_df['Pred_Att']
            actual_col = results_df['Actual_Att']
        elif target == 'Yards':
            pred_col = results_df['Pred_Yds']
            actual_col = results_df['Actual_Yds']
        else:  # Touchdowns
            pred_col = results_df['Pred_TD']
            actual_col = results_df['Actual_TD']
        
        ax.scatter(actual_col, pred_col, alpha=0.6, s=30)
        
        # Perfect prediction line
        min_val = min(min(actual_col), min(pred_col))
        max_val = max(max(actual_col), max(pred_col))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Trend line
        z = np.polyfit(actual_col, pred_col, 1)
        p = np.poly1d(z)
        ax.plot(actual_col, p(actual_col), "b-", alpha=0.8, linewidth=2, label='Trend Line')
        
        ax.set_xlabel(f'Actual {target}')
        ax.set_ylabel(f'Predicted {target}')
        ax.set_title(f'{target}: R² = {metrics[target]["R²"]:.3f}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 8. Error Distribution (Yards)
    ax8 = plt.subplot(3, 4, 8)
    yards_errors = results_df['Yds_Error']
    ax8.hist(yards_errors, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax8.set_title('Yards Error Distribution', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Absolute Error (Yards)')
    ax8.set_ylabel('Frequency')
    ax8.axvline(yards_errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {yards_errors.mean():.1f}')
    ax8.legend()
    
    # 9. Performance by Year
    ax9 = plt.subplot(3, 4, 9)
    yearly_performance = results_df.groupby('Year').agg({
        'Yds_Error': 'mean',
        'Att_Error': 'mean',
        'TD_Error': 'mean'
    }).reset_index()
    
    ax9.plot(yearly_performance['Year'], yearly_performance['Yds_Error'], marker='o', label='Yards MAE', linewidth=2)
    ax9.plot(yearly_performance['Year'], yearly_performance['Att_Error'], marker='s', label='Attempts MAE', linewidth=2)
    ax9.plot(yearly_performance['Year'], yearly_performance['TD_Error'], marker='^', label='TD MAE', linewidth=2)
    ax9.set_title('Performance Trends by Year', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Year')
    ax9.set_ylabel('Mean Absolute Error')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. MAPE Comparison
    ax10 = plt.subplot(3, 4, 10)
    mape_scores = [metrics[target]['MAPE'] for target in targets]
    bars = ax10.bar(targets, mape_scores, color=['#9467bd', '#8c564b', '#e377c2'])
    ax10.set_title('Mean Absolute Percentage Error', fontsize=14, fontweight='bold')
    ax10.set_ylabel('MAPE (%)')
    for i, v in enumerate(mape_scores):
        ax10.text(i, v + max(mape_scores) * 0.02, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 11. Bias Analysis
    ax11 = plt.subplot(3, 4, 11)
    bias_scores = [metrics[target]['Relative_Bias_%'] for target in targets]
    colors = ['red' if x < 0 else 'green' for x in bias_scores]
    bars = ax11.bar(targets, bias_scores, color=colors, alpha=0.7)
    ax11.set_title('Relative Bias (%)', fontsize=14, fontweight='bold')
    ax11.set_ylabel('Bias (%)')
    ax11.axhline(y=0, color='black', linestyle='-', linewidth=1)
    for i, v in enumerate(bias_scores):
        ax11.text(i, v + (1 if v >= 0 else -1), f'{v:.1f}%', ha='center', 
                 va='bottom' if v >= 0 else 'top', fontweight='bold')
    
    # 12. Model Performance Summary
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    summary_text = f"""
MODEL PERFORMANCE SUMMARY

Overall Metrics:
• Average R²: {metrics['Overall']['Average_R²']:.4f}
• Average MAE: {metrics['Overall']['Average_MAE']:.2f}
• Average Correlation: {metrics['Overall']['Average_Correlation']:.4f}
• Total Predictions: {metrics['Overall']['Total_Predictions']:,}

Best Performance:
• Yards R²: {metrics['Yards']['R²']:.4f}
• Attempts R²: {metrics['Attempts']['R²']:.4f}
• Touchdowns R²: {metrics['Touchdowns']['R²']:.4f}

Accuracy:
• Yards within 20%: {metrics['Yards']['Within_20%']:.1f}%
• Attempts within 20%: {metrics['Attempts']['Within_20%']:.1f}%
• TD within 20%: {metrics['Touchdowns']['Within_20%']:.1f}%
"""
    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout(pad=3.0)
    
    if save_plots:
        plt.savefig(DATA_DIR / 'comprehensive_model_performance.png', dpi=300, bbox_inches='tight')
        print("  ✅ Performance visualizations saved to comprehensive_model_performance.png")
    
    plt.show()
    
    return fig

def generate_performance_report(metrics, results_df):
    """Generate a comprehensive performance report."""
    print("\n📋 GENERATING COMPREHENSIVE PERFORMANCE REPORT")
    print("=" * 60)
    
    report = []
    report.append("NFL RUSHING PREDICTION MODEL - PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append(f"Training Period: 1990-2010")
    report.append(f"Testing Period: 2010-2024")
    report.append(f"Total Predictions: {metrics['Overall']['Total_Predictions']:,}")
    report.append(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 20)
    report.append(f"• Overall Model R²: {metrics['Overall']['Average_R²']:.4f}")
    report.append(f"• Overall Correlation: {metrics['Overall']['Average_Correlation']:.4f}")
    report.append(f"• Best Performing Target: Yards (R² = {metrics['Yards']['R²']:.4f})")
    report.append(f"• Model shows {'strong' if metrics['Overall']['Average_R²'] > 0.5 else 'moderate' if metrics['Overall']['Average_R²'] > 0.3 else 'weak'} predictive power")
    report.append("")
    
    # Detailed Metrics by Target
    for target in ['Attempts', 'Yards', 'Touchdowns']:
        report.append(f"{target.upper()} PREDICTION PERFORMANCE")
        report.append("-" * 30)
        report.append(f"• R² Score: {metrics[target]['R²']:.4f}")
        report.append(f"• Mean Absolute Error: {metrics[target]['MAE']:.2f}")
        report.append(f"• Root Mean Square Error: {metrics[target]['RMSE']:.2f}")
        report.append(f"• Mean Absolute Percentage Error: {metrics[target]['MAPE']:.1f}%")
        report.append(f"• Pearson Correlation: {metrics[target]['Pearson_Correlation']:.4f}")
        report.append(f"• Spearman Correlation: {metrics[target]['Spearman_Correlation']:.4f}")
        report.append(f"• Predictions within 10%: {metrics[target]['Within_10%']:.1f}%")
        report.append(f"• Predictions within 20%: {metrics[target]['Within_20%']:.1f}%")
        report.append(f"• Predictions within 30%: {metrics[target]['Within_30%']:.1f}%")
        report.append(f"• Relative Bias: {metrics[target]['Relative_Bias_%']:.1f}%")
        report.append(f"• Mean Actual: {metrics[target]['Mean_Actual']:.1f}")
        report.append(f"• Mean Predicted: {metrics[target]['Mean_Predicted']:.1f}")
        report.append("")
    
    # Model Strengths and Weaknesses
    report.append("MODEL ANALYSIS")
    report.append("-" * 15)
    
    # Identify strengths
    strengths = []
    if metrics['Yards']['R²'] > 0.4:
        strengths.append("Strong yards prediction capability")
    if metrics['Overall']['Average_Correlation'] > 0.6:
        strengths.append("High correlation between predictions and actuals")
    if metrics['Yards']['Within_20%'] > 60:
        strengths.append("Good accuracy for practical applications")
    
    report.append("Strengths:")
    for strength in strengths if strengths else ["Model shows baseline predictive capability"]:
        report.append(f"• {strength}")
    
    # Identify weaknesses
    weaknesses = []
    if metrics['Touchdowns']['R²'] < 0.3:
        weaknesses.append("Touchdown prediction challenging (high variance)")
    if metrics['Attempts']['MAPE'] > 25:
        weaknesses.append("Attempt prediction has high percentage error")
    if abs(metrics['Yards']['Relative_Bias_%']) > 10:
        weaknesses.append("Systematic bias in yards predictions")
    
    report.append("\nWeaknesses:")
    for weakness in weaknesses if weaknesses else ["No major systematic weaknesses identified"]:
        report.append(f"• {weakness}")
    
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 15)
    report.append("• Focus on yards predictions for fantasy/betting applications")
    report.append("• Use ensemble approach for robustness")
    report.append("• Consider additional features for TD prediction improvement")
    report.append("• Regular model retraining recommended")
    report.append("• Monitor for concept drift in modern NFL")
    report.append("")
    
    # Statistical Significance
    report.append("STATISTICAL SIGNIFICANCE")
    report.append("-" * 25)
    for target in ['Attempts', 'Yards', 'Touchdowns']:
        p_val = metrics[target]['Pearson_P_Value']
        significance = "Highly Significant" if p_val < 0.001 else "Significant" if p_val < 0.05 else "Not Significant"
        report.append(f"• {target} Correlation: {significance} (p = {p_val:.6f})")
    
    # Save report
    report_text = "\n".join(report)
    report_file = DATA_DIR / "comprehensive_performance_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print("  ✅ Performance report saved to comprehensive_performance_report.txt")
    print("\nKEY FINDINGS:")
    print(f"  • Overall R²: {metrics['Overall']['Average_R²']:.4f}")
    print(f"  • Yards R²: {metrics['Yards']['R²']:.4f}")
    print(f"  • Best Accuracy: {max(metrics['Yards']['Within_20%'], metrics['Attempts']['Within_20%'], metrics['Touchdowns']['Within_20%']):.1f}% within 20%")
    
    return report_text

def main():
    """Main execution function with comprehensive backtesting."""
    print("🏈 NFL Comprehensive Rushing Predictor with Advanced Backtesting")
    print("=" * 80)
    print("Training: 1990-2010 | Testing: 2010-2024")
    print("Features: 62 comprehensive metrics | Models: 5 ensemble algorithms")
    print()
    
    # Load and merge data
    data = load_and_merge_data()
    if data is None:
        return
    
    # Run comprehensive backtesting
    metrics, results_df = comprehensive_backtest(
        data, 
        train_years=(1990, 2010), 
        test_years=(2010, 2024)
    )
    
    if metrics is None or results_df is None:
        print("❌ Backtesting failed. Check your data files and try again.")
        return
    
    # Create performance visualizations
    try:
        fig = create_performance_visualizations(metrics, results_df, save_plots=True)
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")
        print("Continuing with text-based results...")
    
    # Generate comprehensive performance report
    report_text = generate_performance_report(metrics, results_df)
    
    # Save detailed results
    if not results_df.empty:
        results_file = DATA_DIR / "comprehensive_backtest_results_1990_2024.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n💾 Detailed results saved to {results_file}")
        
        # Save metrics summary
        metrics_summary = []
        for target in ['Attempts', 'Yards', 'Touchdowns']:
            metrics_summary.append({
                'Target': target,
                'R²': metrics[target]['R²'],
                'MAE': metrics[target]['MAE'],
                'RMSE': metrics[target]['RMSE'],
                'MAPE': metrics[target]['MAPE'],
                'Pearson_Correlation': metrics[target]['Pearson_Correlation'],
                'Pearson_P_Value': metrics[target]['Pearson_P_Value'],
                'Within_20%': metrics[target]['Within_20%'],
                'Relative_Bias_%': metrics[target]['Relative_Bias_%']
            })
        
        metrics_df = pd.DataFrame(metrics_summary)
        metrics_file = DATA_DIR / "model_performance_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"💾 Performance metrics saved to {metrics_file}")
    
    # Display sample results
    print("\n🔍 SAMPLE PREDICTION RESULTS")
    print("=" * 100)
    print(f"{'Player':<18} {'Year':<5} {'Age':<3} {'Predicted (A/Y/TD)':<18} {'Actual (A/Y/TD)':<16} {'Errors':<12}")
    print("-" * 100)
    
    # Show best and worst predictions
    results_df['Total_Error'] = results_df['Att_Error'] + results_df['Yds_Error'] + results_df['TD_Error']
    
    print("BEST PREDICTIONS:")
    best_predictions = results_df.nsmallest(10, 'Total_Error')
    for _, row in best_predictions.iterrows():
        pred_str = f"{row['Pred_Att']}/{row['Pred_Yds']}/{row['Pred_TD']}"
        actual_str = f"{row['Actual_Att']}/{row['Actual_Yds']}/{row['Actual_TD']}"
        error_str = f"{row['Att_Error']}/{row['Yds_Error']}/{row['TD_Error']}"
        print(f"{row['Player'][:17]:<18} {row['Year']:<5} {row['Age']:<3} {pred_str:<18} {actual_str:<16} {error_str}")
    
    print("\nWORST PREDICTIONS:")
    worst_predictions = results_df.nlargest(10, 'Total_Error')
    for _, row in worst_predictions.iterrows():
        pred_str = f"{row['Pred_Att']}/{row['Pred_Yds']}/{row['Pred_TD']}"
        actual_str = f"{row['Actual_Att']}/{row['Actual_Yds']}/{row['Actual_TD']}"
        error_str = f"{row['Att_Error']}/{row['Yds_Error']}/{row['TD_Error']}"
        print(f"{row['Player'][:17]:<18} {row['Year']:<5} {row['Age']:<3} {pred_str:<18} {actual_str:<16} {error_str}")
    
    # Year-by-year performance summary
    print(f"\n📈 YEAR-BY-YEAR PERFORMANCE SUMMARY")
    print("=" * 70)
    yearly_summary = results_df.groupby('Year').agg({
        'Pred_Yds': 'count',
        'Yds_Error': 'mean',
        'Att_Error': 'mean', 
        'TD_Error': 'mean',
        'Yds_Error_Pct': 'mean'
    }).round(1)
    yearly_summary.columns = ['Predictions', 'Yds_MAE', 'Att_MAE', 'TD_MAE', 'Yds_MAPE%']
    print(yearly_summary)
    
    # Feature importance analysis (if we have the feature names)
    print(f"\n🎯 MODEL INSIGHTS")
    print("=" * 40)
    print(f"• Training Data Period: 1990-2010 ({1990}-{2010})")
    print(f"• Testing Data Period: 2010-2024 ({2010+1}-{2024})")
    print(f"• Total Test Predictions: {len(results_df):,}")
    print(f"• Average Annual Predictions: {len(results_df)/(2024-2010):.0f}")
    print(f"• Best R² Score: {max(metrics['Attempts']['R²'], metrics['Yards']['R²'], metrics['Touchdowns']['R²']):.4f}")
    print(f"• Most Predictable: {'Yards' if metrics['Yards']['R²'] == max(metrics['Attempts']['R²'], metrics['Yards']['R²'], metrics['Touchdowns']['R²']) else 'Attempts' if metrics['Attempts']['R²'] == max(metrics['Attempts']['R²'], metrics['Yards']['R²'], metrics['Touchdowns']['R²']) else 'Touchdowns'}")
    
    # Performance tier analysis
    print(f"\n🏆 PERFORMANCE TIER ANALYSIS")
    print("=" * 35)
    
    def get_performance_tier(r2_score):
        if r2_score >= 0.6:
            return "Excellent"
        elif r2_score >= 0.4:
            return "Good"
        elif r2_score >= 0.2:
            return "Fair"
        else:
            return "Poor"
    
    for target in ['Attempts', 'Yards', 'Touchdowns']:
        tier = get_performance_tier(metrics[target]['R²'])
        print(f"• {target}: {tier} (R² = {metrics[target]['R²']:.4f})")
    
    overall_tier = get_performance_tier(metrics['Overall']['Average_R²'])
    print(f"• Overall Model: {overall_tier} (Avg R² = {metrics['Overall']['Average_R²']:.4f})")
    
    print(f"\n🎉 COMPREHENSIVE BACKTESTING COMPLETED!")
    print(f"Check the generated files for detailed analysis:")
    print(f"  • comprehensive_backtest_results_1990_2024.csv")
    print(f"  • model_performance_metrics.csv") 
    print(f"  • comprehensive_performance_report.txt")
    print(f"  • comprehensive_model_performance.png")

if __name__ == "__main__":
    main()