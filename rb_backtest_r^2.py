#!/usr/bin/env python3
"""
evaluate_rushing_model.py

1) Train on 2011–2023 top-10 rushers (loaded from your CSVs),
   predicting each player's next-season stats via full-table fetch.
2) Back-test on seasons 2000–2009 → 2001–2010 (nlargest top-10).
3) Report MAE, MAPE, and R² and save detailed predictions/errors.
"""

import os
import time
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR = "."   # where top10_rushing_<year>.csv live
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}

def fetch_rushing_table(year: int, max_retries: int = 5) -> pd.DataFrame:
    """
    Fetch and clean the full rushing table for a season, with
    exponential backoff on HTTP 429.
    """
    url = f"https://www.pro-football-reference.com/years/{year}/rushing.htm"
    backoff = 5
    for attempt in range(1, max_retries+1):
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code == 429:
            print(f"[{year}] 429 – backing off {backoff}s (attempt {attempt})")
            time.sleep(backoff); backoff *= 2
            continue
        resp.raise_for_status()
        df = pd.read_html(resp.text)[0]
        df.columns = df.columns.droplevel(0)
        df = df[df.Rk != "Rk"].drop(columns="Rk").reset_index(drop=True)
        for col in ['Age','G','GS','Att','Yds','TD','Lng','Y/A','Y/G','Fmb']:
            if col in df: df[col] = pd.to_numeric(df[col], errors='coerce')
        time.sleep(3)
        return df
    raise RuntimeError(f"Failed to fetch {year} after {max_retries} retries")

def load_top10_csv(year: int) -> pd.DataFrame:
    """
    Load your pre-saved top10_rushing_<year>.csv.
    """
    path = os.path.join(DATA_DIR, f"top10_rushing_{year}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    # ensure numeric
    for col in ['Att','Yds','TD']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df

def build_historical_training(start_year=2010, end_year=2024):
    """
    Use exactly the 10 players in top10 CSVs for each year N to predict their
    full-table stats in N+1.
    """
    X_rows, y_rows = [], []
    for yr in range(start_year, end_year):
        df_n   = load_top10_csv(yr)                  # exactly 10 players
        full_n1 = fetch_rushing_table(yr+1)          # need next-year full table
        for _, r in df_n.iterrows():
            nxt = full_n1[full_n1['Player']==r['Player']]
            if nxt.empty:
                continue
            X_rows.append([r['Att'], r['Yds'], r['TD']])
            a = nxt.iloc[0]
            y_rows.append([int(a['Att']), int(a['Yds']), int(a['TD'])])
    X = pd.DataFrame(X_rows, columns=['Att_N','Yds_N','TD_N'])
    y = pd.DataFrame(y_rows, columns=['Att_N1','Yds_N1','TD_N1'])
    return X, y

def train_model(X, y):
    """
    Fit a multi-output linear regression.
    """
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X, y)
    return model

def backtest(model, start_year=2000, end_year=2010):
    """
    For seasons 2000–2009: fetch top-10 by Att, predict N+1,
    compare to actual full-table stats in N+1.
    """
    recs = []
    for yr in range(start_year, end_year):
        top_n   = fetch_rushing_table(yr).nlargest(10, 'Att').reset_index(drop=True)
        full_n1 = fetch_rushing_table(yr+1)
        Xn = top_n[['Att','Yds','TD']].rename(
            columns={'Att':'Att_N','Yds':'Yds_N','TD':'TD_N'}
        )
        preds = model.predict(Xn)
        for idx, r in top_n.iterrows():
            player = r['Player']
            pred_att, pred_yds, pred_td = preds[idx]
            actual = full_n1[full_n1['Player']==player]
            if actual.empty:
                continue
            a = actual.iloc[0]
            recs.append({
                'Season_N': yr,
                'Player':   player,
                'Att_N':    int(r['Att']),
                'Yds_N':    int(r['Yds']),
                'TD_N':     int(r['TD']),
                'Att_pred': round(pred_att),
                'Yds_pred': round(pred_yds),
                'TD_pred':  round(pred_td),
                'Att_act':  int(a['Att']),
                'Yds_act':  int(a['Yds']),
                'TD_act':   int(a['TD']),
            })

    df_preds = pd.DataFrame(recs)
    metrics = {}
    for stat in ['Att','Yds','TD']:
        mae  = mean_absolute_error(df_preds[f'{stat}_act'], df_preds[f'{stat}_pred'])
        mape = (abs(df_preds[f'{stat}_act'] - df_preds[f'{stat}_pred']) /
                df_preds[f'{stat}_act']).mean() * 100
        r2   = r2_score(df_preds[f'{stat}_act'], df_preds[f'{stat}_pred'])
        
        metrics[f'{stat}_MAE']  = mae
        metrics[f'{stat}_MAPE'] = mape
        metrics[f'{stat}_R2']   = r2

    df_err = pd.DataFrame([metrics])
    return df_preds, df_err

def main():
    print("1) Building training data from your 2010–2023 top-10 CSVs…")
    X, y = build_historical_training(2011, 2024)
    print(f"   → {len(X)} samples (exactly 10 per year)")

    print("2) Training multi-output linear regression…")
    model = train_model(X, y)

    print("3) Back-testing on seasons 2000–2009 → 2001–2010…")
    df_back, df_err = backtest(model, 2000, 2010)

    print("\nAggregate error metrics (2001–2010):")
    print(df_err.to_string(index=False))

    # Save outputs
    df_back.to_csv("backtest_rushing_preds.csv", index=False)
    df_err.to_csv("backtest_rushing_errors.csv", index=False)
    print("\nSaved detailed preds → backtest_rushing_preds.csv")
    print("Saved error metrics → backtest_rushing_errors.csv")

if __name__ == "__main__":
    main()