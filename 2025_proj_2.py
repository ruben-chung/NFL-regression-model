from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class WRProjectionModel:
    def __init__(self, data_dir="nfl_wr_data"):
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.models = {}
        self.feature_importance = {}

    def load_data(self):
        receiving_file = self.data_dir / "/Users/rubenchung/Desktop/GitHUB/nfl_wr_data/all_receiving_data.csv"
        receiving_df = pd.read_csv(receiving_file)

        rushing_file = self.data_dir / "/Users/rubenchung/Desktop/GitHUB/nfl_wr_data/all_rushing_data.csv"
        rushing_df = pd.read_csv(rushing_file) if rushing_file.exists() else pd.DataFrame()

        team_file = self.data_dir / "/Users/rubenchung/Desktop/GitHUB/nfl_wr_data/all_team_stats.csv"
        team_df = pd.read_csv(team_file) if team_file.exists() else pd.DataFrame()

        df = receiving_df.copy()
        if not rushing_df.empty:
            merge_cols = ['Player', 'Year']
            rush_cols = merge_cols + [col for col in rushing_df.columns if col.startswith('Rush_')]
            df = df.merge(rushing_df[rush_cols], on=merge_cols, how='left')

        if not team_df.empty:
            team_cols = ['Tm', 'Year', 'PF', 'PA', 'TOT', 'Ply', 'Y/P']
            available_team_cols = [col for col in team_cols if col in team_df.columns]
            if len(available_team_cols) >= 2:
                df = df.merge(team_df[available_team_cols], on=['Tm', 'Year'], how='left')

        self.raw_data = df
        return df

    def engineer_base_features(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        if 'Age' in df.columns:
            df['Age_Squared'] = df['Age'] ** 2
            df['Prime_Age'] = ((df['Age'] >= 25) & (df['Age'] <= 29)).astype(int)
            df['Rookie_Sophomore'] = (df['Age'] <= 23).astype(int)

        if 'Tgt' in df.columns and 'Rec' in df.columns:
            df['Target_Share'] = df['Tgt'] / df['G'].clip(lower=1)
            df['Catch_Rate'] = df['Rec'] / df['Tgt'].clip(lower=1)

        if 'Yds' in df.columns:
            df['Yards_Per_Game'] = df['Yds'] / df['G'].clip(lower=1)

        if 'TD' in df.columns and 'Rec' in df.columns:
            df['TD_Per_Reception'] = df['TD'] / df['Rec'].clip(lower=1)

        if 'TD' in df.columns and 'Tgt' in df.columns:
            df['TD_Per_Target'] = df['TD'] / df['Tgt'].clip(lower=1)

        if 'Y/R' in df.columns:
            df['High_YPR'] = (df['Y/R'] > 12).astype(int)

        if 'PF' in df.columns:
            df['Team_Offense_Strength'] = df['PF'] / df['G'].clip(lower=1)

        df['Games_Played_Rate'] = df['G'] / 16

        return df

    def engineer_temporal_features(self, df, max_year=None):
        df = df.sort_values(['Player', 'Year'])
        if max_year is not None:
            feature_data = df[df['Year'] <= max_year].copy()
        else:
            feature_data = df.copy()

        feature_data['Experience'] = feature_data.groupby('Player').cumcount()

        lag_cols = ['Tgt', 'Rec', 'Yds', 'TD', 'Y/R']
        for col in lag_cols:
            if col in feature_data.columns:
                feature_data[f'{col}_Prev'] = feature_data.groupby('Player')[col].shift(1)

        rolling_cols = ['TD', 'Yds', 'Rec', 'Tgt']
        for col in rolling_cols:
            if col in feature_data.columns:
                feature_data[f'{col}_Avg2'] = (
                    feature_data.groupby('Player')[col].rolling(2, min_periods=1).mean().reset_index(0, drop=True)
                )

        temporal_cols = ['Experience'] + [f'{col}_Prev' for col in lag_cols if col in df.columns] + \
                        [f'{col}_Avg2' for col in rolling_cols if col in df.columns]
        merge_keys = ['Player', 'Year']
        temporal_features = feature_data[merge_keys + temporal_cols]
        result_df = df.merge(temporal_features, on=merge_keys, how='left')

        for col in temporal_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(0)

        return result_df

    def prepare_modeling_data(self, df, target_cols=['TD']):
        exclude_cols = ['Player', 'Tm', 'Pos', 'Year']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols + target_cols]
        df_model = df[(df['G'] >= 4) & (df['Tgt'] >= 10)].copy()
        X = df_model[feature_cols]
        targets = {target: df_model[target] for target in target_cols if target in df_model.columns}
        self.feature_names = feature_cols
        return X, targets, df_model

    def train_models(self, X_train, y_train_dict):
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_train_final = pd.DataFrame(X_train_scaled, columns=self.feature_names, index=X_train.index)
        model = LinearRegression()

        for target_name, y_train in y_train_dict.items():
            self.models[target_name] = {}
            model.fit(X_train_final, y_train)
            cv_scores = cross_val_score(model, X_train_final, y_train, cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            self.models[target_name]['Linear'] = {'model': model, 'cv_mae': cv_mae, 'cv_std': cv_scores.std()}
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': model.coef_,
                'abs_coefficient': np.abs(model.coef_)
            }).sort_values('abs_coefficient', ascending=False)
            self.feature_importance[target_name] = importance_df

    def evaluate_models(self, X_test, y_test_dict):
        X_test_imputed = self.imputer.transform(X_test)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        X_test_final = pd.DataFrame(X_test_scaled, columns=self.feature_names, index=X_test.index)

        results = {}
        for target_name, y_test in y_test_dict.items():
            model_info = self.models[target_name]['Linear']
            model = model_info['model']
            y_pred = model.predict(X_test_final)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            results[target_name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
        self.test_results = results
        return results

    def create_projections(self, current_year_data):
        X_current = current_year_data[self.feature_names]
        X_current_imputed = self.imputer.transform(X_current)
        X_current_scaled = self.scaler.transform(X_current_imputed)
        td_model = self.models['TD']['Linear']['model']
        td_proj = td_model.predict(X_current_scaled)
        proj_df = current_year_data[['Player', 'Tm', 'Age', 'G']].copy()
        proj_df['TD_Proj'] = td_proj
        proj_df['TD_Per_Game'] = td_proj / proj_df['G'].clip(lower=1)
        proj_df['TD_Tier'] = pd.cut(proj_df['TD_Proj'], bins=[0, 3, 6, 10, 100],
                                    labels=['Low (0-3)', 'Medium (4-6)', 'High (7-10)', 'Elite (10+)'])
        return proj_df.sort_values('TD_Proj', ascending=False)

    def generate_projection_input_for_next_season(self, df, next_year=2025):
        last_years = df.groupby("Player")["Year"].max().reset_index()
        last_data = df.merge(last_years, on=["Player", "Year"], how="inner")
        last_data = last_data[last_data["Year"] == next_year - 1]
        proj_input = last_data.copy()
        proj_input["Year"] = next_year
        for col in ["TD"]:
            if col in proj_input.columns:
                proj_input = proj_input.drop(columns=[col])
        return proj_input


def main():
    model = WRProjectionModel()
    df = model.load_data()
    df_base = model.engineer_base_features(df)
    df_with_temporal = model.engineer_temporal_features(df_base, max_year=2010)

    X, y_dict, model_data = model.prepare_modeling_data(df_with_temporal)
    train_mask = model_data['Year'] <= 2010
    test_mask = model_data['Year'] >= 2011

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train_dict = {target: y[train_mask] for target, y in y_dict.items()}
    y_test_dict = {target: y[test_mask] for target, y in y_dict.items()}

    model.train_models(X_train, y_train_dict)
    model.evaluate_models(X_test, y_test_dict)

    # Generate 2025 projections
    df_proj_input = model.generate_projection_input_for_next_season(model.raw_data, next_year=2025)
    df_proj_input = model.engineer_base_features(df_proj_input)
    df_proj_input = model.engineer_temporal_features(df_proj_input, max_year=2024)
    df_proj_input = df_proj_input[(df_proj_input['G'] >= 4) & (df_proj_input['Tgt'] >= 10)]

    if df_proj_input.empty:
        print("⚠️ No players eligible for 2025 projection.")
    else:
        proj_df = model.create_projections(df_proj_input)
        print("\n🏆 Top 10 Projected WR TD Scorers for 2025:")
        print(proj_df[['Player', 'Tm', 'Age', 'G', 'TD_Proj', 'TD_Per_Game', 'TD_Tier']].head(10).to_string(index=False))

    return model


if __name__ == "__main__":
    main()
