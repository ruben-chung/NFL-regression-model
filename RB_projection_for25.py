#!/usr/bin/env python3
"""
train_predict_rushing.py

Train a multi-output linear regression on NFL top-10 rushers
(2010–2023) to predict next-season performance, then project
2025 stats for the 2024 top-10 group.
"""

import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

DATA_DIR = "."  # where top10_rushing_<year>.csv files live

def load_top10(year):
    path = os.path.join(DATA_DIR, f"top10_rushing_{year}.csv")
    return pd.read_csv(path)

def build_training_data(start_year=2011, end_year=2024):
    records = []
    for year in range(start_year, end_year):
        df_n  = load_top10(year)
        df_n1 = load_top10(year+1)
        for _, row in df_n.iterrows():
            player = row['Player']
            nxt = df_n1[df_n1['Player']==player]
            if nxt.empty:
                continue
            records.append({
                'Att_N':  row['Att'],
                'Yds_N':  row['Yds'],
                'TD_N':   row['TD'],
                'Att_N1': int(nxt.iloc[0]['Att']),
                'Yds_N1': int(nxt.iloc[0]['Yds']),
                'TD_N1':  int(nxt.iloc[0]['TD']),
            })
    df = pd.DataFrame(records)
    X  = df[['Att_N','Yds_N','TD_N']]
    y  = df[['Att_N1','Yds_N1','TD_N1']]
    return X, y

def train_model(X, y):
    """
    Fit a multi-output linear regression.
    """
    regr = MultiOutputRegressor(LinearRegression())
    regr.fit(X, y)
    return regr

def predict_next_season(model, df_current, season_label):
    """
    Given a DataFrame for season N, predict N+1 stats.
    """
    # Rename to match feature names used in training:
    X_cur = (
        df_current[['Att','Yds','TD']]
        .rename(columns={'Att':'Att_N', 'Yds':'Yds_N', 'TD':'TD_N'})
    )
    preds = model.predict(X_cur)
    df_out = df_current[['Player']].copy()
    df_out[f'Att_{season_label}'] = preds[:,0].round().astype(int)
    df_out[f'Yds_{season_label}'] = preds[:,1].round().astype(int)
    df_out[f'TD_{season_label}']  = preds[:,2].round().astype(int)
    return df_out

def main():
    # 1) Build training data from 2010→2023 → targets 2011→2024
    X, y   = build_training_data(2011, 2024)
    # 2) Train
    model  = train_model(X, y)
    # 3) Load 2024 top-10 and predict 2025
    df24   = load_top10(2024)
    df25   = predict_next_season(model, df24, season_label=2025)
    # 4) Output
    print("\nProjected 2025 stats for 2024 Top-10 Rushers:\n")
    print(df25.to_string(index=False))
    df25.to_csv("proj_rushing_2025.csv", index=False)
    print("\nSaved projections to proj_rushing_2025.csv")

if __name__ == "__main__":
    main()
