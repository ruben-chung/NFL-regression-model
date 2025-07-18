import requests
import pandas as pd
import time
from bs4 import BeautifulSoup, Comment

def fetch_pfr_table(year: int, stat: str) -> pd.DataFrame:
    """
    Fetches https://www.pro-football-reference.com/years/{year}/{stat}.htm,
    extracts the commented table with id="{stat}", and returns it as a DataFrame.
    """
    url = f"https://www.pro-football-reference.com/years/{year}/{stat}.htm"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    # PFR wraps these tables in HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if f'id="{stat}"' in comment:
            tbl = BeautifulSoup(comment, "lxml").find("table", id=stat)
            df = pd.read_html(str(tbl))[0]
            # drop any header rows repeated in the body
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
            df = df[df['Rk'] != 'Rk']
            return df
    raise ValueError(f"Table '{stat}' not found for {year}")

def build_rb_stats(start_year=2005, end_year=2024, pause=1.0) -> pd.DataFrame:
    records = []
    for yr in range(start_year, end_year + 1):
        print(f"→ Season {yr}")
        # 1) get rushing and receiving tables
        rush = fetch_pfr_table(yr, 'rushing')
        recv = fetch_pfr_table(yr, 'receiving')

        # 2) keep only RBs and numeric-convert
        rush = rush[rush['Pos']=='RB'].copy()
        recv = recv[recv['Pos']=='RB'].copy()
        rush['Yds'] = pd.to_numeric(rush['Yds'], errors='coerce')
        rush['TD']  = pd.to_numeric(rush['TD'],  errors='coerce')
        recv['Rec'] = pd.to_numeric(recv['Rec'], errors='coerce')
        recv['Yds'] = pd.to_numeric(recv['Yds'], errors='coerce')
        recv['TD']  = pd.to_numeric(recv['TD'],  errors='coerce')

        # 3) aggregate across teams (in case of mid‐season trades)
        rush_agg = rush.groupby('Player')[['Yds','TD']].sum().rename(
            columns={'Yds':'rush_yds','TD':'rush_td'}
        )
        recv_agg = recv.groupby('Player')[['Rec','Yds','TD']].sum().rename(
            columns={'Rec':'receptions','Yds':'rec_yds','TD':'rec_td'}
        )

        # 4) merge & compute total TDs
        stats = rush_agg.join(recv_agg, how='outer').fillna(0)
        stats['total_tds'] = stats['rush_td'] + stats['rec_td']
        stats = stats.reset_index().rename(columns={'Player':'player'})

        # 5) record season and append
        stats['season'] = yr
        records.append(stats)

        time.sleep(pause)  # polite scraping

    # 6) concat all seasons, filter to RB1 each year
    all_stats = pd.concat(records, ignore_index=True)
    # to get RB1 per season, sort by total fantasy points (optional) or by rush_yds
    # here we pick the back with highest (rush_yds + rec_yds) just as an example
    all_stats['scrimmage_yds'] = all_stats['rush_yds'] + all_stats['rec_yds']
    rb1 = (all_stats
           .sort_values(['season','scrimmage_yds'], ascending=[True, False])
           .groupby('season')
           .first()
           .reset_index()
           [['season','player','rush_yds','receptions','total_tds']])
    return rb1

if __name__ == "__main__":
    df_rb1 = build_rb_stats(2005, 2024, pause=1.5)
    print(df_rb1.to_string(index=False))
    df_rb1.to_csv("rb1_by_season.csv", index=False)
    print("\nSaved → rb1_by_season.csv")
