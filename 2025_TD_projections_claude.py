"""
wr_ml_projections_no_leakage.py

Machine Learning model to project WR TDs with ZERO data leakage.
Uses ONLY historical data to predict future performance.
Train on 1992-2010, backtest on 2011-2024.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WRProjectionModelNoLeakage:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.models = {}
        self.feature_importance = {}
        self.feature_names = []
        
    def load_and_merge_data(self):
        """Load and merge all datasets with proper column handling."""
        print("📊 Loading WR datasets...")
        
        # Load receiving data
        receiving_df = pd.read_csv('/Users/rubenchung/Desktop/GitHUB/nfl_wr_data/all_receiving_data.csv')
        print(f"  ✅ Loaded {len(receiving_df)} receiving records")
        
        # Load rushing data for WRs (if available)
        try:
            rushing_df = pd.read_csv('all_rushing_data.csv')
            # Merge on Player and Year only
            df = receiving_df.merge(rushing_df[['Player', 'Year', 'Rush_Att', 'Rush_Yds', 'Rush_TD']], 
                                  on=['Player', 'Year'], how='left')
            print(f"  ✅ Merged rushing data")
        except:
            df = receiving_df
            print(f"  ⚠️ No rushing data found")
        
        # Load team stats (if available)
        try:
            team_df = pd.read_csv('all_team_stats.csv')
            # Merge team context (previous year team stats to avoid leakage)
            df = df.merge(team_df[['Tm', 'Year', 'PA']], 
                         on=['Tm', 'Year'], how='left', suffixes=('', '_team'))
            print(f"  ✅ Merged team context")
        except:
            print(f"  ⚠️ No team data found")
        
        print(f"  📊 Final dataset: {len(df)} records")
        return df
    
    def create_historical_features_only(self, df):
        """Create ONLY historical features with strict temporal boundaries."""
        print("🔒 Creating historical features (NO current year data)...")
        
        # Sort by player and year
        df = df.sort_values(['Player', 'Year']).reset_index(drop=True)
        
        # Fill missing values
        numeric_cols = ['Age', 'G', 'GS', 'Tgt', 'Rec', 'Yds', 'TD', 'Fmb']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # ONLY use lagged (previous year) features
        print("  📈 Creating lag features (previous year only)...")
        
        # 1. Previous year performance
        lag_features = ['Tgt', 'Rec', 'Yds', 'TD', 'G', 'Age']
        for feature in lag_features:
            if feature in df.columns:
                df[f'{feature}_Lag1'] = df.groupby('Player')[feature].shift(1)
        
        # 2. Two years ago performance  
        for feature in lag_features:
            if feature in df.columns:
                df[f'{feature}_Lag2'] = df.groupby('Player')[feature].shift(2)
        
        # 3. Career averages (EXCLUDING current year)
        print("  📊 Creating career averages (historical only)...")
        career_features = ['Tgt', 'Rec', 'Yds', 'TD']
        for feature in career_features:
            if feature in df.columns:
                # Expanding mean of previous years only
                shifted_data = df.groupby('Player')[feature].shift(1)
                df[f'{feature}_Career_Avg'] = shifted_data.groupby(df['Player']).expanding().mean().reset_index(0, drop=True)
        
        # 4. Experience (years in league)
        df['Experience'] = df.groupby('Player').cumcount()
        
        # 5. Age-based indicators (these don't leak)
        if 'Age' in df.columns:
            df['Prime_Age'] = ((df['Age'] >= 24) & (df['Age'] <= 29)).astype(int)
            df['Young_Player'] = (df['Age'] <= 23).astype(int)
            df['Veteran'] = (df['Age'] >= 30).astype(int)
        
        # 6. Historical trends (change from 2 years ago to 1 year ago)
        trend_features = ['Tgt', 'Rec', 'Yds', 'TD']
        for feature in trend_features:
            lag1_col = f'{feature}_Lag1'
            lag2_col = f'{feature}_Lag2'
            if lag1_col in df.columns and lag2_col in df.columns:
                # Year-over-year change (using only historical data)
                df[f'{feature}_Trend'] = np.where(
                    df[lag2_col] > 0, 
                    (df[lag1_col] - df[lag2_col]) / df[lag2_col], 
                    0
                )
        
        # 7. Historical consistency (standard deviation of past performances)
        for feature in ['TD', 'Tgt']:
            if feature in df.columns:
                # Rolling std of previous years only
                shifted_data = df.groupby('Player')[feature].shift(1)
                df[f'{feature}_Historical_Std'] = shifted_data.groupby(df['Player']).rolling(3, min_periods=2).std().reset_index(0, drop=True)
        
        # 8. Team context (use previous year team performance to avoid leakage)
        if 'PA' in df.columns:
            df['Team_Defense_Lag1'] = df.groupby('Tm')['PA'].shift(1)
        
        # Clean up - replace infinite values and fill NaN
        feature_cols = [col for col in df.columns if any(suffix in col for suffix in 
                       ['_Lag1', '_Lag2', '_Career_Avg', '_Trend', '_Historical_Std', 'Experience', 'Prime_Age', 'Young_Player', 'Veteran', 'Team_Defense_Lag1'])]
        
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
        
        print(f"  ✅ Created {len(feature_cols)} pure historical features")
        print(f"  🚫 ZERO current-year features used")
        
        return df, feature_cols
    
    def prepare_train_test_split(self, df, feature_cols):
        """Prepare proper train/test split with NO data leakage."""
        print("🔄 Creating leak-proof train/test split...")
        
        # Filter to players with sufficient history
        # Must have at least some previous year data to make predictions
        df_filtered = df.dropna(subset=[col for col in feature_cols if 'Lag1' in col], how='all')
        df_filtered = df_filtered[(df_filtered['Tgt'] >= 10) & (df_filtered['G'] >= 4)]
        
        print(f"  📊 Filtered to {len(df_filtered)} records with historical data")
        
        # Strict temporal split: Train 1992-2010, Test 2011-2024
        train_data = df_filtered[df_filtered['Year'] <= 2010].copy()
        test_data = df_filtered[df_filtered['Year'] >= 2011].copy()
        
        print(f"  📊 Train: {len(train_data)} records (1992-2010)")
        print(f"  📊 Test: {len(test_data)} records (2011-2024)")
        
        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError("No data available for train/test split!")
        
        # Prepare features and targets
        X_train = train_data[feature_cols]
        X_test = test_data[feature_cols]
        y_train = train_data['TD']
        y_test = test_data['TD']
        
        # Handle missing values and scale
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_test_imputed = self.imputer.transform(X_test)
        
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        # Convert back to DataFrames
        X_train_final = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        X_test_final = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        
        self.feature_names = feature_cols
        
        print(f"  ✅ Features: {len(feature_cols)} historical predictors")
        print(f"  🎯 Target range - Train: {y_train.min():.0f} to {y_train.max():.0f} TDs")
        print(f"  🎯 Target range - Test: {y_test.min():.0f} to {y_test.max():.0f} TDs")
        
        return X_train_final, X_test_final, y_train, y_test, train_data, test_data
    
    def train_models(self, X_train, y_train):
        """Train multiple models."""
        print("🤖 Training models for TD prediction...")
        
        models_config = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        }
        
        for model_name, model in models_config.items():
            print(f"  🔧 Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            
            self.models[model_name] = {
                'model': model,
                'cv_mae': cv_mae,
                'cv_std': cv_scores.std()
            }
            
            print(f"    ✅ {model_name}: CV MAE = {cv_mae:.2f} ± {cv_scores.std():.2f}")
        
        # Feature importance for Random Forest
        if 'RandomForest' in self.models:
            rf_model = self.models['RandomForest']['model']
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"  🔍 Top 5 predictive features:")
            for _, row in self.feature_importance.head().iterrows():
                print(f"    • {row['feature']}: {row['importance']:.3f}")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models on test set."""
        print(f"\n📊 Evaluating models on 2011-2024 data...")
        
        results = {}
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'predictions': y_pred
            }
            
            print(f"  {model_name}:")
            print(f"    MAE: {mae:.2f} TDs")
            print(f"    RMSE: {rmse:.2f} TDs")
            print(f"    R²: {r2:.3f}")
        
        self.test_results = results
        return results
    
    def plot_results(self):
        """Create visualizations of model performance."""
        print("📈 Creating model performance visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('WR TD Prediction - NO DATA LEAKAGE Results', fontsize=14, fontweight='bold')
        
        # Model comparison - MAE
        ax1 = axes[0, 0]
        models = list(self.test_results.keys())
        mae_scores = [self.test_results[m]['MAE'] for m in models]
        r2_scores = [self.test_results[m]['R²'] for m in models]
        
        bars1 = ax1.bar(models, mae_scores, alpha=0.7, color=['blue', 'green', 'orange'])
        ax1.set_title('Model Comparison (MAE)')
        ax1.set_ylabel('Mean Absolute Error (TDs)')
        
        for bar, mae in zip(bars1, mae_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{mae:.2f}', ha='center', va='bottom')
        
        # Model comparison - R²
        ax2 = axes[0, 1]
        bars2 = ax2.bar(models, r2_scores, alpha=0.7, color=['blue', 'green', 'orange'])
        ax2.set_title('Model Comparison (R²)')
        ax2.set_ylabel('R² Score')
        
        for bar, r2 in zip(bars2, r2_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{r2:.3f}', ha='center', va='bottom')
        
        # Feature importance (Random Forest)
        if hasattr(self, 'feature_importance') and not self.feature_importance.empty:
            ax3 = axes[1, 0]
            top_features = self.feature_importance.head(8)
            bars3 = ax3.barh(top_features['feature'], top_features['importance'])
            ax3.set_title('Top Predictive Features (Random Forest)')
            ax3.set_xlabel('Importance')
            ax3.invert_yaxis()
        
        # Actual vs Predicted (best model)
        ax4 = axes[1, 1]
        best_model = min(self.test_results.items(), key=lambda x: x[1]['MAE'])[0]
        
        ax4.text(0.5, 0.5, f'Best Model: {best_model}\n\n'
                          f'R² = {self.test_results[best_model]["R²"]:.3f}\n'
                          f'MAE = {self.test_results[best_model]["MAE"]:.2f}\n\n'
                          f'Expected R² Range:\n0.10 - 0.35\n(Realistic for predicting\nfuture TD performance)',
                ha='center', va='center', transform=ax4.transAxes, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                fontsize=10)
        ax4.set_title('Model Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def create_2025_projections(self):
        """Create 2025 projections using 2024 data."""
        print(f"\n🔮 Creating 2025 TD projections...")
        
        # Load and process all data
        df = self.load_and_merge_data()
        df_with_features, feature_cols = self.create_historical_features_only(df)
        
        # Get 2024 data for projections
        data_2024 = df_with_features[df_with_features['Year'] == 2024].copy()
        
        # Filter to active players with historical data
        active_players = data_2024.dropna(subset=[col for col in feature_cols if 'Lag1' in col], how='all')
        active_players = active_players[(active_players['Tgt'] >= 20) & (active_players['G'] >= 8)]
        
        if active_players.empty:
            print("❌ No active players found for 2025 projections")
            return pd.DataFrame()
        
        print(f"  📊 Found {len(active_players)} active WRs for 2025 projections")
        
        # Prepare features
        X_2025 = active_players[feature_cols]
        X_2025_imputed = self.imputer.transform(X_2025)
        X_2025_scaled = self.scaler.transform(X_2025_imputed)
        
        # Get predictions from best model
        best_model_name = min(self.test_results.items(), key=lambda x: x[1]['MAE'])[0]
        model = self.models[best_model_name]['model']
        
        td_proj_2025 = model.predict(X_2025_scaled)
        
        # Create projections dataframe
        proj_2025 = pd.DataFrame({
            'Player': active_players['Player'].values,
            'Team': active_players['Tm'].values,
            'Age': active_players['Age'].values,
            '2024_TDs': active_players['TD'].values,
            '2025_TD_Projection': np.maximum(td_proj_2025, 0),  # No negative TDs
            'TD_Change': td_proj_2025 - active_players['TD'].values,
            'Model_Used': best_model_name
        })
        
        # Add categories
        proj_2025['Change_Category'] = pd.cut(
            proj_2025['TD_Change'], 
            bins=[-100, -1.5, -0.5, 0.5, 1.5, 100],
            labels=['Big Decline', 'Small Decline', 'Similar', 'Small Increase', 'Big Increase']
        )
        
        proj_2025 = proj_2025.sort_values('2025_TD_Projection', ascending=False)
        
        print(f"  ✅ 2025 projections created using {best_model_name} model")
        print(f"  📊 Average projected TDs: {proj_2025['2025_TD_Projection'].mean():.1f}")
        
        return proj_2025
    
    def plot_2025_projections(self, proj_2025):
        """Visualize 2025 projections."""
        if proj_2025.empty:
            return
        
        print("📊 Creating 2025 projection visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('2025 WR TD Projections (No Data Leakage)', fontsize=16, fontweight='bold')
        
        # Scatter plot: 2024 vs 2025 projected
        ax1 = axes[0, 0]
        colors = {'Big Decline': 'red', 'Small Decline': 'orange', 'Similar': 'gray', 
                 'Small Increase': 'lightgreen', 'Big Increase': 'green'}
        
        for category in colors.keys():
            mask = proj_2025['Change_Category'] == category
            if mask.any():
                ax1.scatter(proj_2025.loc[mask, '2024_TDs'], 
                           proj_2025.loc[mask, '2025_TD_Projection'],
                           c=colors[category], label=category, alpha=0.7, s=50)
        
        # Add diagonal line
        max_td = max(proj_2025['2024_TDs'].max(), proj_2025['2025_TD_Projection'].max())
        ax1.plot([0, max_td], [0, max_td], 'k--', alpha=0.5, label='No Change')
        ax1.set_xlabel('2024 Actual TDs')
        ax1.set_ylabel('2025 Projected TDs')
        ax1.set_title('2024 vs 2025 TD Projections')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Top projected performers
        ax2 = axes[0, 1]
        top_10 = proj_2025.head(10)
        bars = ax2.barh(range(len(top_10)), top_10['2025_TD_Projection'], alpha=0.7)
        ax2.set_yticks(range(len(top_10)))
        ax2.set_yticklabels(top_10['Player'])
        ax2.set_xlabel('Projected 2025 TDs')
        ax2.set_title('Top 10 Projected TD Scorers')
        ax2.invert_yaxis()
        
        # Biggest increases
        ax3 = axes[1, 0]
        increases = proj_2025.nlargest(8, 'TD_Change')
        bars = ax3.barh(range(len(increases)), increases['TD_Change'], color='green', alpha=0.7)
        ax3.set_yticks(range(len(increases)))
        ax3.set_yticklabels(increases['Player'])
        ax3.set_xlabel('Projected TD Increase')
        ax3.set_title('Biggest Projected Increases')
        ax3.invert_yaxis()
        
        # Biggest decreases
        ax4 = axes[1, 1]
        decreases = proj_2025.nsmallest(8, 'TD_Change')
        bars = ax4.barh(range(len(decreases)), decreases['TD_Change'], color='red', alpha=0.7)
        ax4.set_yticks(range(len(decreases)))
        ax4.set_yticklabels(decreases['Player'])
        ax4.set_xlabel('Projected TD Change')
        ax4.set_title('Biggest Projected Decreases')
        ax4.invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        # Print key insights
        self._print_2025_insights(proj_2025)
    
    def _print_2025_insights(self, proj_2025):
        """Print key insights from 2025 projections."""
        print(f"\n🏆 2025 TD PROJECTION INSIGHTS")
        print("="*50)
        
        print(f"\n🚀 TOP BREAKOUT CANDIDATES:")
        breakouts = proj_2025.nlargest(5, 'TD_Change')
        for _, player in breakouts.iterrows():
            if player['TD_Change'] > 0:
                print(f"  {player['Player']:20} - {player['2024_TDs']:.0f} → {player['2025_TD_Projection']:.1f} TDs ({player['TD_Change']:+.1f})")
        
        print(f"\n📉 REGRESSION CANDIDATES:")
        regressions = proj_2025.nsmallest(5, 'TD_Change')
        for _, player in regressions.iterrows():
            if player['TD_Change'] < 0:
                print(f"  {player['Player']:20} - {player['2024_TDs']:.0f} → {player['2025_TD_Projection']:.1f} TDs ({player['TD_Change']:+.1f})")
        
        print(f"\n🎯 TOP 2025 PROJECTIONS:")
        top_proj = proj_2025.head(8)
        for _, player in top_proj.iterrows():
            print(f"  {player['Player']:20} - {player['2025_TD_Projection']:.1f} projected TDs")
    
    def run_analysis(self):
        """Run the complete analysis."""
        print("🏈 WR TD PREDICTION - ZERO DATA LEAKAGE VERSION")
        print("="*60)
        
        # Load and process data
        df = self.load_and_merge_data()
        df_with_features, feature_cols = self.create_historical_features_only(df)
        
        # Train/test split
        X_train, X_test, y_train, y_test, train_data, test_data = self.prepare_train_test_split(df_with_features, feature_cols)
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_models(X_test, y_test)
        
        # Visualize results
        self.plot_results()
        
        # Create 2025 projections
        proj_2025 = self.create_2025_projections()
        if not proj_2025.empty:
            self.plot_2025_projections(proj_2025)
            
            # Save projections
            proj_2025.to_csv('wr_2025_td_projections_no_leakage.csv', index=False)
            print(f"\n💾 Projections saved to: wr_2025_td_projections_no_leakage.csv")
        
        # Final summary
        best_model = min(results.items(), key=lambda x: x[1]['MAE'])[0]
        print(f"\n🎉 ANALYSIS COMPLETE - NO DATA LEAKAGE!")
        print(f"✅ Best model: {best_model}")
        print(f"✅ Test R²: {results[best_model]['R²']:.3f} (realistic range: 0.10-0.35)")
        print(f"✅ Test MAE: {results[best_model]['MAE']:.2f} TDs")
        print(f"✅ Uses ONLY historical data to predict future TDs")
        print(f"✅ Proper temporal validation prevents overfitting")
        
        return results, proj_2025

def main():
    """Run the leak-proof analysis."""
    model = WRProjectionModelNoLeakage()
    results, proj_2025 = model.run_analysis()
    return model, results, proj_2025

if __name__ == "__main__":
    model, results, proj_2025 = main()