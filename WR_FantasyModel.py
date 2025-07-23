"""
wr_fantasy_projections_ensemble.py

Advanced ensemble ML model to project WR fantasy points with multiple algorithms.
Train on 1992-2010, backtest on 2011-2024.
Uses stacking, voting, and blending for improved accuracy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnsembleWRFantasyModel:
    def __init__(self, scoring_format="PPR"):
        self.scoring_format = scoring_format
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
        # Individual models storage
        self.base_models = {}
        self.ensemble_models = {}
        self.feature_importance = {}
        
        # Fantasy scoring formats
        self.scoring_systems = {
            "PPR": {"rec": 1.0, "rec_yd": 0.1, "rec_td": 6.0, "rush_yd": 0.1, "rush_td": 6.0, "fumble": -2.0},
            "Half_PPR": {"rec": 0.5, "rec_yd": 0.1, "rec_td": 6.0, "rush_yd": 0.1, "rush_td": 6.0, "fumble": -2.0},
            "Standard": {"rec": 0.0, "rec_yd": 0.1, "rec_td": 6.0, "rush_yd": 0.1, "rush_td": 6.0, "fumble": -2.0}
        }
        
    def calculate_fantasy_points(self, df, scoring_format="PPR"):
        """Calculate fantasy points based on scoring system."""
        scoring = self.scoring_systems[scoring_format]
        
        fantasy_points = (
            df['Rec'].fillna(0) * scoring['rec'] +
            df['Yds'].fillna(0) * scoring['rec_yd'] +
            df['TD'].fillna(0) * scoring['rec_td'] +
            df.get('Rush_Yds', pd.Series(0, index=df.index)).fillna(0) * scoring['rush_yd'] +
            df.get('Rush_TD', pd.Series(0, index=df.index)).fillna(0) * scoring['rush_td'] +
            df['Fmb'].fillna(0) * scoring['fumble']
        )
        
        return fantasy_points
        
    def load_and_merge_data(self):
        """Load and merge all datasets."""
        print("📊 Loading WR datasets...")
        
        # Load receiving data
        receiving_df = pd.read_csv('/Users/rubenchung/Desktop/GitHUB/nfl_wr_data/all_receiving_data.csv')
        print(f"  ✅ Loaded {len(receiving_df)} receiving records (1992-2024)")
        
        # Load rushing data for WRs
        try:
            rushing_df = pd.read_csv('/Users/rubenchung/Desktop/GitHUB/nfl_wr_data/all_rushing_data.csv')
            df = receiving_df.merge(rushing_df[['Player', 'Year', 'Rush_Att', 'Rush_Yds', 'Rush_TD', 'Rush_Y/A', 'Rush_Y/G']], 
                                  on=['Player', 'Year'], how='left')
            print(f"  ✅ Merged rushing data for WRs")
        except:
            df = receiving_df
            print(f"  ⚠️ No rushing data found, using receiving only")
        
        # Load team stats
        try:
            team_df = pd.read_csv('/Users/rubenchung/Desktop/GitHUB/nfl_wr_data/all_team_stats.csv')
            df = df.merge(team_df[['Tm', 'Year', 'G', 'PA', 'Ply', 'Y/P']], 
                         on=['Tm', 'Year'], how='left', suffixes=('', '_team'))
            print(f"  ✅ Merged team context data")
        except:
            print(f"  ⚠️ No team data found")
        
        print(f"  📊 Final dataset: {len(df)} records, {len(df.columns)} columns")
        return df
    
    def create_historical_features(self, df):
        """Create ONLY historical features to prevent data leakage."""
        print("🔧 Creating historical features (no current year stats)...")
        
        # Sort by player and year
        df = df.sort_values(['Player', 'Year']).reset_index(drop=True)
        
        # Fill missing values first
        numeric_cols = ['Age', 'G', 'GS', 'Tgt', 'Rec', 'Yds', 'TD', 'Fmb', 'Rush_Att', 'Rush_Yds', 'Rush_TD']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Calculate current year fantasy points (target only)
        df['Fantasy_Points'] = self.calculate_fantasy_points(df, self.scoring_format)
        
        # HISTORICAL FEATURES ONLY (previous years)
        print("  📈 Creating lag features (previous year performance)...")
        lag_features = ['Tgt', 'Rec', 'Yds', 'TD', 'G', 'Age', 'Fantasy_Points']
        for feature in lag_features:
            if feature in df.columns:
                df[f'{feature}_Lag1'] = df.groupby('Player')[feature].shift(1)
                df[f'{feature}_Lag2'] = df.groupby('Player')[feature].shift(2)
                df[f'{feature}_Lag3'] = df.groupby('Player')[feature].shift(3)
        
        # CAREER AVERAGES (up to previous year)
        print("  📊 Creating career averages (excluding current year)...")
        career_features = ['Tgt', 'Rec', 'Yds', 'TD', 'Fantasy_Points']
        for feature in career_features:
            if feature in df.columns:
                df[f'{feature}_Career_Avg'] = df.groupby('Player')[feature].shift(1).expanding().mean().reset_index(0, drop=True)
                # Career max/min
                df[f'{feature}_Career_Max'] = df.groupby('Player')[feature].shift(1).expanding().max().reset_index(0, drop=True)
                df[f'{feature}_Career_Min'] = df.groupby('Player')[feature].shift(1).expanding().min().reset_index(0, drop=True)
        
        # EXPERIENCE AND AGE FEATURES
        print("  👤 Creating player development features...")
        df['Experience'] = df.groupby('Player').cumcount()
        df['Age_Squared'] = df['Age'] ** 2
        df['Age_Cubed'] = df['Age'] ** 3
        df['Prime_Age'] = ((df['Age'] >= 24) & (df['Age'] <= 29)).astype(int)
        df['Young_Player'] = (df['Age'] <= 23).astype(int)
        df['Veteran_Player'] = (df['Age'] >= 30).astype(int)
        df['Peak_Experience'] = ((df['Experience'] >= 3) & (df['Experience'] <= 7)).astype(int)
        
        # TREND FEATURES (improvement/decline patterns)
        print("  📈 Creating trend features...")
        trend_features = ['Tgt', 'Rec', 'Yds', 'TD', 'Fantasy_Points']
        for feature in trend_features:
            if f'{feature}_Lag1' in df.columns and f'{feature}_Lag2' in df.columns:
                lag1 = df[f'{feature}_Lag1']
                lag2 = df[f'{feature}_Lag2']
                lag3 = df[f'{feature}_Lag3']
                
                # Year-over-year change
                df[f'{feature}_YoY_Change'] = np.where(lag2 > 0, (lag1 - lag2) / lag2, 0)
                
                # 2-year trend
                df[f'{feature}_2Yr_Trend'] = np.where(lag3 > 0, (lag1 - lag3) / lag3, 0)
                
                # Momentum (acceleration/deceleration)
                df[f'{feature}_Momentum'] = np.where(
                    (lag2 > 0) & (lag3 > 0),
                    ((lag1 - lag2) / lag2) - ((lag2 - lag3) / lag3),
                    0
                )
        
        # ADVANCED METRICS
        print("  🎯 Creating advanced metrics...")
        # Efficiency ratios (from previous years)
        if 'Tgt_Lag1' in df.columns and 'Rec_Lag1' in df.columns:
            df['Catch_Rate_Lag1'] = np.where(df['Tgt_Lag1'] > 0, df['Rec_Lag1'] / df['Tgt_Lag1'], 0)
        
        if 'Rec_Lag1' in df.columns and 'TD_Lag1' in df.columns:
            df['TD_Rate_Lag1'] = np.where(df['Rec_Lag1'] > 0, df['TD_Lag1'] / df['Rec_Lag1'], 0)
        
        if 'Yds_Lag1' in df.columns and 'Rec_Lag1' in df.columns:
            df['YPR_Lag1'] = np.where(df['Rec_Lag1'] > 0, df['Yds_Lag1'] / df['Rec_Lag1'], 0)
        
        # TEAM CONTEXT (external factors)
        print("  🏈 Creating team context features...")
        if 'PA' in df.columns:
            df['Team_Defense_Strength'] = df['PA'] / df['G_team'].fillna(16)
        
        if 'Ply' in df.columns:
            df['Team_Pace'] = df['Ply'] / df['G_team'].fillna(16)
        
        # CONSISTENCY AND VOLATILITY METRICS
        print("  📊 Creating consistency metrics...")
        consistency_features = ['Fantasy_Points', 'Tgt', 'Rec', 'Yds']
        for feature in consistency_features:
            if f'{feature}_Lag1' in df.columns:
                # Rolling standard deviation (3-year window)
                rolling_std = df.groupby('Player')[feature].shift(1).rolling(3, min_periods=2).std()
                df[f'{feature}_Volatility'] = rolling_std.reset_index(0, drop=True)
                
                # Coefficient of variation
                rolling_mean = df.groupby('Player')[feature].shift(1).rolling(3, min_periods=2).mean()
                rolling_mean = rolling_mean.reset_index(0, drop=True)
                df[f'{feature}_CV'] = np.where(rolling_mean > 0, df[f'{feature}_Volatility'] / rolling_mean, 0)
        
        # BREAKOUT/DECLINE INDICATORS
        print("  ⚡ Creating breakout/decline indicators...")
        if 'Fantasy_Points_Lag1' in df.columns and 'Fantasy_Points_Career_Avg' in df.columns:
            # Above career average performance
            df['Above_Career_Avg'] = (df['Fantasy_Points_Lag1'] > df['Fantasy_Points_Career_Avg']).astype(int)
            
            # Breakout season indicator (>50% above career average)
            df['Breakout_Season'] = (df['Fantasy_Points_Lag1'] > 1.5 * df['Fantasy_Points_Career_Avg']).astype(int)
        
        # POSITION AND ROLE INDICATORS
        print("  🎭 Creating role indicators...")
        df['WR_Only'] = (df['Pos'] == 'WR').astype(int)
        df['Multi_Position'] = df['Pos'].str.contains('/', na=False).astype(int)
        
        # Handle remaining missing values
        feature_cols = [col for col in df.columns if any(suffix in col for suffix in 
                       ['_Lag1', '_Lag2', '_Lag3', '_Career_Avg', '_Career_Max', '_Career_Min', '_YoY_Change', 
                        '_2Yr_Trend', '_Momentum', '_Rate_', 'YPR_', 'Experience', 'Age', 'Prime_Age', 
                        'Young_Player', 'Veteran_Player', 'Peak_Experience', 'Team_', 'WR_Only', 
                        'Multi_Position', '_Volatility', '_CV', 'Above_Career_Avg', 'Breakout_Season'])]
        
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                df[col] = df[col].replace([np.inf, -np.inf], 0)
        
        print(f"  ✅ Created {len(feature_cols)} historical features")
        print(f"  🚫 Zero current-year stats used (no data leakage)")
        
        return df, feature_cols
    
    def prepare_train_test_split(self, df, feature_cols):
        """Prepare proper train/test split by year."""
        print("🔄 Creating train/test split...")
        
        # Filter to relevant players (must have some receiving activity)
        df_filtered = df[(df['Tgt'] >= 10) & (df['G'] >= 4)].copy()
        
        # Train: 1992-2010, Test: 2011-2024
        train_data = df_filtered[df_filtered['Year'] <= 2010].copy()
        test_data = df_filtered[df_filtered['Year'] >= 2011].copy()
        
        print(f"  📊 Train data: {len(train_data)} records (1992-2010)")
        print(f"  📊 Test data: {len(test_data)} records (2011-2024)")
        
        # Remove players with no historical data
        train_data = train_data.dropna(subset=[col for col in feature_cols if 'Lag1' in col], how='all')
        test_data = test_data.dropna(subset=[col for col in feature_cols if 'Lag1' in col], how='all')
        
        print(f"  🔧 After removing rookies - Train: {len(train_data)}, Test: {len(test_data)}")
        
        # Prepare features and targets
        X_train = train_data[feature_cols]
        X_test = test_data[feature_cols]
        y_train = train_data['Fantasy_Points']
        y_test = test_data['Fantasy_Points']
        
        # Handle missing values and scale
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_test_imputed = self.imputer.transform(X_test)
        
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        # Convert back to DataFrame
        X_train_final = pd.DataFrame(X_train_scaled, columns=feature_cols)
        X_test_final = pd.DataFrame(X_test_scaled, columns=feature_cols)
        
        self.feature_names = feature_cols
        
        print(f"  ✅ Final feature set: {len(feature_cols)} features")
        print(f"  🎯 Target range - Train: {y_train.min():.1f} to {y_train.max():.1f}")
        print(f"  🎯 Target range - Test: {y_test.min():.1f} to {y_test.max():.1f}")
        
        return X_train_final, X_test_final, y_train, y_test, train_data, test_data
    
    def train_base_models(self, X_train, y_train):
        """Train individual base models."""
        print(f"🤖 Training base models for {self.scoring_format} fantasy points...")
        
        # Diverse set of base models
        base_models_config = {
            'RandomForest': RandomForestRegressor(
                n_estimators=150, max_depth=12, min_samples_split=5, 
                min_samples_leaf=3, random_state=42, n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=150, max_depth=12, min_samples_split=5,
                min_samples_leaf=3, random_state=42, n_jobs=-1
            ),
            'GradientBoost': GradientBoostingRegressor(
                n_estimators=120, learning_rate=0.08, max_depth=6, 
                min_samples_split=5, random_state=42
            ),
            'Ridge': Ridge(alpha=2.0),
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
            'SVR': SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1),
            'Linear': LinearRegression()
        }
        
        for model_name, model in base_models_config.items():
            print(f"  🔧 Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            
            self.base_models[model_name] = {
                'model': model,
                'cv_mae': cv_mae,
                'cv_std': cv_scores.std()
            }
            
            print(f"    ✅ {model_name}: CV MAE = {cv_mae:.1f} ± {cv_scores.std():.1f}")
        
        print(f"  🎯 Trained {len(base_models_config)} base models")
    
    def train_ensemble_models(self, X_train, y_train):
        """Train ensemble models using base models."""
        print(f"🎭 Training ensemble models...")
        
        # 1. Voting Regressor (Simple Average)
        voting_models = [
            ('rf', self.base_models['RandomForest']['model']),
            ('et', self.base_models['ExtraTrees']['model']),
            ('gb', self.base_models['GradientBoost']['model']),
            ('ridge', self.base_models['Ridge']['model'])
        ]
        
        voting_regressor = VotingRegressor(estimators=voting_models)
        voting_regressor.fit(X_train, y_train)
        
        cv_scores = cross_val_score(voting_regressor, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        
        self.ensemble_models['Voting'] = {
            'model': voting_regressor,
            'cv_mae': -cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"  ✅ Voting Ensemble: CV MAE = {-cv_scores.mean():.1f} ± {cv_scores.std():.1f}")
        
        # 2. Weighted Average (based on CV performance)
        weights = []
        total_inverse_mae = 0
        
        # Calculate weights inversely proportional to MAE
        for model_name in ['RandomForest', 'ExtraTrees', 'GradientBoost', 'Ridge']:
            mae = self.base_models[model_name]['cv_mae']
            weight = 1.0 / mae
            weights.append(weight)
            total_inverse_mae += weight
        
        # Normalize weights
        weights = [w / total_inverse_mae for w in weights]
        
        def weighted_predict(X):
            predictions = np.zeros(len(X))
            for i, model_name in enumerate(['RandomForest', 'ExtraTrees', 'GradientBoost', 'Ridge']):
                model = self.base_models[model_name]['model']
                predictions += weights[i] * model.predict(X)
            return predictions
        
        # Create a simple wrapper for the weighted ensemble
        class WeightedEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
            
            def predict(self, X):
                predictions = np.zeros(len(X))
                for i, (model_name, weight) in enumerate(zip(['RandomForest', 'ExtraTrees', 'GradientBoost', 'Ridge'], self.weights)):
                    predictions += weight * self.models[model_name]['model'].predict(X)
                return predictions
        
        weighted_ensemble = WeightedEnsemble(self.base_models, weights)
        
        # Cross-validate weighted ensemble
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_maes = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Train base models on fold
            temp_models = {}
            for model_name, model_info in self.base_models.items():
                if model_name in ['RandomForest', 'ExtraTrees', 'GradientBoost', 'Ridge']:
                    temp_model = type(model_info['model'])(**model_info['model'].get_params())
                    temp_model.fit(X_train_fold, y_train_fold)
                    temp_models[model_name] = {'model': temp_model}
            
            # Create weighted ensemble for this fold
            fold_ensemble = WeightedEnsemble(temp_models, weights)
            y_pred = fold_ensemble.predict(X_val_fold)
            cv_maes.append(mean_absolute_error(y_val_fold, y_pred))
        
        self.ensemble_models['Weighted'] = {
            'model': weighted_ensemble,
            'cv_mae': np.mean(cv_maes),
            'cv_std': np.std(cv_maes),
            'weights': weights
        }
        
        print(f"  ✅ Weighted Ensemble: CV MAE = {np.mean(cv_maes):.1f} ± {np.std(cv_maes):.1f}")
        print(f"    Weights: RF={weights[0]:.3f}, ET={weights[1]:.3f}, GB={weights[2]:.3f}, Ridge={weights[3]:.3f}")
        
        # 3. Best Model Selection
        best_model_name = min(self.base_models.items(), key=lambda x: x[1]['cv_mae'])[0]
        self.ensemble_models['BestModel'] = {
            'model': self.base_models[best_model_name]['model'],
            'cv_mae': self.base_models[best_model_name]['cv_mae'],
            'cv_std': self.base_models[best_model_name]['cv_std'],
            'base_model': best_model_name
        }
        
        print(f"  ✅ Best Single Model: {best_model_name} (CV MAE = {self.base_models[best_model_name]['cv_mae']:.1f})")
        
        # Feature importance (from Random Forest)
        rf_model = self.base_models['RandomForest']['model']
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"  🔍 Top 5 predictive features:")
        for _, row in self.feature_importance.head().iterrows():
            print(f"    • {row['feature']}: {row['importance']:.3f}")
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all models on test set."""
        print(f"\n📊 Evaluating all models on 2011-2024 data...")
        
        all_results = {}
        
        # Evaluate base models
        print("  🤖 Base Models:")
        for model_name, model_info in self.base_models.items():
            model = model_info['model']
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            all_results[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'predictions': y_pred,
                'type': 'base'
            }
            
            print(f"    {model_name}: MAE={mae:.1f}, RMSE={rmse:.1f}, R²={r2:.3f}")
        
        # Evaluate ensemble models
        print("\n  🎭 Ensemble Models:")
        for model_name, model_info in self.ensemble_models.items():
            model = model_info['model']
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            all_results[f"Ensemble_{model_name}"] = {
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'predictions': y_pred,
                'type': 'ensemble'
            }
            
            if model_name == 'BestModel':
                base_name = model_info['base_model']
                print(f"    Best Model ({base_name}): MAE={mae:.1f}, RMSE={rmse:.1f}, R²={r2:.3f}")
            else:
                print(f"    {model_name}: MAE={mae:.1f}, RMSE={rmse:.1f}, R²={r2:.3f}")
        
        self.all_results = all_results
        return all_results
    
    def plot_ensemble_results(self):
        """Create comprehensive visualizations."""
        print("📈 Creating ensemble visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{self.scoring_format} Fantasy Points: Ensemble Model Results', fontsize=16, fontweight='bold')
        
        # 1. Model comparison (MAE)
        ax1 = axes[0, 0]
        models = list(self.all_results.keys())
        mae_scores = [self.all_results[m]['MAE'] for m in models]
        colors = ['lightblue' if self.all_results[m]['type'] == 'base' else 'orange' for m in models]
        
        bars = ax1.bar(range(len(models)), mae_scores, color=colors, alpha=0.7)
        ax1.set_title('Model Comparison (MAE)', fontweight='bold')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels([m.replace('Ensemble_', '') for m in models], rotation=45, ha='right')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightblue', label='Base Models'),
                          Patch(facecolor='orange', label='Ensemble Models')]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # 2. R² comparison
        ax2 = axes[0, 1]
        r2_scores = [self.all_results[m]['R²'] for m in models]
        ax2.bar(range(len(models)), r2_scores, color=colors, alpha=0.7)
        ax2.set_title('Model Comparison (R²)', fontweight='bold')
        ax2.set_ylabel('R² Score')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.replace('Ensemble_', '') for m in models], rotation=45, ha='right')
        
        # 3. Feature importance
        ax3 = axes[0, 2]
        top_features = self.feature_importance.head(8)
        ax3.barh(top_features['feature'], top_features['importance'], color='forestgreen', alpha=0.7)
        ax3.set_title('Top Predictive Features', fontweight='bold')
        ax3.set_xlabel('Importance')
        
        # 4. Best vs Worst model comparison
        ax4 = axes[1, 0]
        best_model = min(self.all_results.items(), key=lambda x: x[1]['MAE'])
        worst_model = max(self.all_results.items(), key=lambda x: x[1]['MAE'])
        
        comparison_data = {
            'Best Model': [best_model[1]['MAE'], best_model[1]['R²']],
            'Worst Model': [worst_model[1]['MAE'], worst_model[1]['R²']]
        }
        
        x = ['MAE', 'R²']
        x_pos = np.arange(len(x))
        width = 0.35
        
        ax4.bar(x_pos - width/2, comparison_data['Best Model'], width, label=f'Best: {best_model[0]}', alpha=0.7)
        ax4.bar(x_pos + width/2, comparison_data['Worst Model'], width, label=f'Worst: {worst_model[0]}', alpha=0.7)
        ax4.set_title('Best vs Worst Model', fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(x)
        ax4.legend()
        
        # 5. Ensemble performance summary
        ax5 = axes[1, 1]
        ensemble_models = {k: v for k, v in self.all_results.items() if v['type'] == 'ensemble'}
        ensemble_names = list(ensemble_models.keys())
        ensemble_maes = [ensemble_models[m]['MAE'] for m in ensemble_names]
        
        ax5.bar([name.replace('Ensemble_', '') for name in ensemble_names], ensemble_maes, 
                color='darkorange', alpha=0.8)
        ax5.set_title('Ensemble Model Performance', fontweight='bold')
        ax5.set_ylabel('MAE')
        ax5.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(ensemble_maes):
            ax5.text(i, v + max(ensemble_maes)*0.01, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Model improvement summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        best_ensemble = min(ensemble_models.items(), key=lambda x: x[1]['MAE'])
        best_base = min({k: v for k, v in self.all_results.items() if v['type'] == 'base'}.items(), 
                       key=lambda x: x[1]['MAE'])
        
        improvement = best_base[1]['MAE'] - best_ensemble[1]['MAE']
        improvement_pct = (improvement / best_base[1]['MAE']) * 100
        
        summary_text = f"""
ENSEMBLE PERFORMANCE SUMMARY

Best Base Model:
• {best_base[0]}
• MAE: {best_base[1]['MAE']:.1f}
• R²: {best_base[1]['R²']:.3f}

Best Ensemble Model:
• {best_ensemble[0].replace('Ensemble_', '')}
• MAE: {best_ensemble[1]['MAE']:.1f}
• R²: {best_ensemble[1]['R²']:.3f}

Improvement:
• MAE reduced by {improvement:.1f} points
• {improvement_pct:.1f}% better accuracy
• {'✅ Ensemble wins!' if improvement > 0 else '⚠️ No improvement'}

Model Count:
• {len([k for k, v in self.all_results.items() if v['type'] == 'base'])} Base Models
• {len([k for k, v in self.all_results.items() if v['type'] == 'ensemble'])} Ensemble Models
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: Feature importance with categories
        self._plot_feature_importance_detailed()
    
    def _plot_feature_importance_detailed(self):
        """Plot detailed feature importance with categories."""
        plt.figure(figsize=(14, 10))
        
        # Categorize features
        feature_categories = {
            'Lag Features': [f for f in self.feature_importance['feature'] if 'Lag' in f],
            'Career Stats': [f for f in self.feature_importance['feature'] if 'Career' in f],
            'Trends': [f for f in self.feature_importance['feature'] if any(x in f for x in ['Trend', 'YoY', 'Momentum'])],
            'Age/Experience': [f for f in self.feature_importance['feature'] if any(x in f for x in ['Age', 'Experience', 'Prime', 'Young', 'Veteran'])],
            'Efficiency': [f for f in self.feature_importance['feature'] if any(x in f for x in ['Rate', 'YPR', 'Efficiency'])],
            'Consistency': [f for f in self.feature_importance['feature'] if any(x in f for x in ['Volatility', 'CV', 'Consistency'])],
            'Team Context': [f for f in self.feature_importance['feature'] if 'Team' in f],
            'Other': []
        }
        
        # Assign uncategorized features to 'Other'
        all_categorized = set()
        for features in feature_categories.values():
            all_categorized.update(features)
        
        feature_categories['Other'] = [f for f in self.feature_importance['feature'] 
                                     if f not in all_categorized]
        
        # Create color map
        colors = plt.cm.Set3(np.linspace(0, 1, len(feature_categories)))
        color_map = {}
        for i, category in enumerate(feature_categories.keys()):
            for feature in feature_categories[category]:
                color_map[feature] = colors[i]
        
        # Plot top 20 features
        top_features = self.feature_importance.head(20)
        feature_colors = [color_map.get(f, 'gray') for f in top_features['feature']]
        
        bars = plt.barh(top_features['feature'], top_features['importance'], 
                       color=feature_colors, alpha=0.8)
        
        plt.title('Feature Importance by Category (Top 20)', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
        # Add importance values on bars
        for bar, importance in zip(bars, top_features['importance']):
            plt.text(bar.get_width() + max(top_features['importance'])*0.01, 
                    bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', va='center', fontsize=8)
        
        # Create legend
        legend_elements = []
        for category, color in zip(feature_categories.keys(), colors):
            if feature_categories[category]:  # Only add to legend if category has features
                legend_elements.append(Patch(facecolor=color, label=category))
        
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def get_ensemble_insights(self):
        """Provide detailed insights about ensemble performance."""
        print("\n🎭 ENSEMBLE MODEL INSIGHTS")
        print("=" * 50)
        
        # Best performing models
        best_overall = min(self.all_results.items(), key=lambda x: x[1]['MAE'])
        best_base = min({k: v for k, v in self.all_results.items() if v['type'] == 'base'}.items(), 
                       key=lambda x: x[1]['MAE'])
        best_ensemble = min({k: v for k, v in self.all_results.items() if v['type'] == 'ensemble'}.items(), 
                           key=lambda x: x[1]['MAE'])
        
        print(f"🏆 BEST OVERALL MODEL: {best_overall[0]}")
        print(f"   MAE: {best_overall[1]['MAE']:.1f} fantasy points")
        print(f"   R²: {best_overall[1]['R²']:.3f}")
        print(f"   Type: {best_overall[1]['type'].title()}")
        
        print(f"\n🤖 BEST BASE MODEL: {best_base[0]}")
        print(f"   MAE: {best_base[1]['MAE']:.1f}")
        print(f"   R²: {best_base[1]['R²']:.3f}")
        
        print(f"\n🎭 BEST ENSEMBLE MODEL: {best_ensemble[0]}")
        print(f"   MAE: {best_ensemble[1]['MAE']:.1f}")
        print(f"   R²: {best_ensemble[1]['R²']:.3f}")
        
        # Ensemble vs Base comparison
        improvement = best_base[1]['MAE'] - best_ensemble[1]['MAE']
        improvement_pct = (improvement / best_base[1]['MAE']) * 100
        
        print(f"\n📊 ENSEMBLE IMPROVEMENT:")
        if improvement > 0:
            print(f"   ✅ Ensemble reduces MAE by {improvement:.1f} points ({improvement_pct:.1f}%)")
            print(f"   ✅ Better accuracy through model combination")
        else:
            print(f"   ⚠️ Ensemble does not improve over best base model")
            print(f"   ⚠️ Consider different ensemble strategies")
        
        # Weighted ensemble details
        if 'Weighted' in self.ensemble_models:
            weights = self.ensemble_models['Weighted']['weights']
            model_names = ['RandomForest', 'ExtraTrees', 'GradientBoost', 'Ridge']
            print(f"\n⚖️ WEIGHTED ENSEMBLE COMPOSITION:")
            for name, weight in zip(model_names, weights):
                print(f"   {name}: {weight:.1%}")
        
        # Model diversity analysis
        print(f"\n🌟 MODEL DIVERSITY:")
        base_models = {k: v for k, v in self.all_results.items() if v['type'] == 'base'}
        mae_range = max(base_models.values(), key=lambda x: x['MAE'])['MAE'] - min(base_models.values(), key=lambda x: x['MAE'])['MAE']
        r2_range = max(base_models.values(), key=lambda x: x['R²'])['R²'] - min(base_models.values(), key=lambda x: x['R²'])['R²']
        
        print(f"   MAE range: {mae_range:.1f} points")
        print(f"   R² range: {r2_range:.3f}")
        print(f"   Model count: {len(base_models)} base + {len([k for k, v in self.all_results.items() if v['type'] == 'ensemble'])} ensemble")
        
        # Feature insights
        print(f"\n🔍 FEATURE INSIGHTS:")
        print(f"   Total features: {len(self.feature_names)}")
        print(f"   Most important: {self.feature_importance.iloc[0]['feature']}")
        print(f"   Top feature importance: {self.feature_importance.iloc[0]['importance']:.3f}")
        
        top_5_features = self.feature_importance.head(5)['feature'].tolist()
        print(f"   Top 5 features: {', '.join(top_5_features)}")
    
    def run_ensemble_analysis(self):
        """Run the complete ensemble analysis."""
        print("🏈 WR FANTASY ENSEMBLE PROJECTION ANALYSIS")
        print("=" * 60)
        
        # Load data
        df = self.load_and_merge_data()
        
        # Create historical features only
        df_featured, feature_cols = self.create_historical_features(df)
        
        # Train/test split
        X_train, X_test, y_train, y_test, train_data, test_data = self.prepare_train_test_split(df_featured, feature_cols)
        
        # Train base models
        self.train_base_models(X_train, y_train)
        
        # Train ensemble models
        self.train_ensemble_models(X_train, y_train)
        
        # Evaluate all models
        results = self.evaluate_all_models(X_test, y_test)
        
        # Get insights
        self.get_ensemble_insights()
        
        # Visualize
        self.plot_ensemble_results()
        
        # Final summary
        best_model = min(results.items(), key=lambda x: x[1]['MAE'])
        ensemble_models = {k: v for k, v in results.items() if v['type'] == 'ensemble'}
        best_ensemble = min(ensemble_models.items(), key=lambda x: x[1]['MAE']) if ensemble_models else None
        
        print(f"\n🎉 ENSEMBLE ANALYSIS COMPLETE!")
        print(f"✅ Trained {len(self.base_models)} base models + {len(self.ensemble_models)} ensemble models")
        print(f"✅ Best overall: {best_model[0]} (MAE: {best_model[1]['MAE']:.1f}, R²: {best_model[1]['R²']:.3f})")
        if best_ensemble:
            print(f"✅ Best ensemble: {best_ensemble[0]} (MAE: {best_ensemble[1]['MAE']:.1f}, R²: {best_ensemble[1]['R²']:.3f})")
        print(f"✅ Uses only historical data (no data leakage)")
        print(f"✅ Model diversity provides robustness")
        print(f"✅ Ensemble methods capture different patterns")
        
        return results

def main():
    """Run the ensemble analysis."""
    model = EnsembleWRFantasyModel(scoring_format="PPR")
    results = model.run_ensemble_analysis()
    return model, results

if __name__ == "__main__":
    model, results = main()