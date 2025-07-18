import pandas as pd
import requests
import time
from bs4 import BeautifulSoup, Comment

# 1) Create a session with realistic browser headers
session = requests.Session()
session.headers.update({
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.google.com/"
})

def fetch_top50_wrs(year, max_retries=5):
    url = f"https://www.pro-football-reference.com/years/{year}/receiving.htm"
    backoff = 1

    # 2) Retry loop with exponential back-off
    for attempt in range(1, max_retries + 1):
        resp = session.get(url, timeout=10, allow_redirects=True)
        print(f"[{year}] attempt {attempt} → HTTP {resp.status_code}")
        if resp.status_code == 200:
            break
        elif resp.status_code == 429:
            time.sleep(backoff)
            backoff *= 2
        else:
            time.sleep(backoff)
            backoff *= 2
    else:
        raise RuntimeError(f"Failed to fetch season {year} after {max_retries} tries (last status {resp.status_code})")

    # 3) Extract the hidden ‘receiving’ table from HTML comments
    soup = BeautifulSoup(resp.text, "html.parser")
    comment = next(
        (c for c in soup.find_all(string=lambda t:isinstance(t, Comment))
         if 'id="receiving"' in c),
        None
    )
    if comment:
        inner = BeautifulSoup(comment, "html.parser")
        table = inner.find("table", {"id": "receiving"})
        df = pd.read_html(str(table))[0]
    else:
        df = pd.read_html(resp.text, attrs={"id": "receiving"})[0]

    # 4) Filter WRs, take top 50 by Yds
    df = df[df["Pos"]=="WR"].dropna(subset=["Player"]).drop_duplicates("Player")
    top50 = (
        df.nlargest(50, "Yds")
          [["Player","Yds","Rec","Tgt","TD"]]
          .rename(columns={
             "Player": "player_name",
             "Yds":    "yds",
             "Rec":    "rec",
             "Tgt":    "tgts"
          })
          .reset_index(drop=True)
    )

    # 5) Compute years of experience
    top50["years_exp"] = top50["player_name"].apply(lambda nm: get_years_exp(nm, year))
    return top50

def get_years_exp(player_name, current_year, max_retries=3):
    first, last = player_name.split()[0], player_name.split()[-1]
    slug = (last[:5] + first[:2] + "00").lower()
    subdir = last[0].lower()
    url = f"https://www.pro-football-reference.com/players/{subdir}/{slug}.htm"

    backoff = 1
    for attempt in range(1, max_retries + 1):
        resp = session.get(url, timeout=10, allow_redirects=True)
        print(f"  [{player_name}] attempt {attempt} → HTTP {resp.status_code}")
        if resp.status_code == 200:
            break
        time.sleep(backoff)
        backoff *= 2
    else:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    comment = next(
        (c for c in soup.find_all(string=lambda t:isinstance(t, Comment))
         if "Career & Play Stats" in c or 'id="stats"' in c),
        None
    )
    if not comment:
        return None

    stats = pd.read_html(str(BeautifulSoup(comment, "html.parser")))[0]
    years = pd.to_numeric(stats["Year"], errors="coerce").dropna().astype(int)
    rookie = int(years.min()) if not years.empty else None
    return (current_year - rookie) if rookie else None

if __name__ == "__main__":
    season_dfs = {}
    for yr in range(2010, 2025):
        print(f"\n=== Fetching top 50 WRs for {yr} ===")
        season_dfs[yr] = fetch_top50_wrs(yr)
        time.sleep(5)   # single pause between seasons

    # Example: inspect 2024
    print(season_dfs[2024].head())

    # Save to CSV
    for yr, df in season_dfs.items():
        path = f"wr_top50_{yr}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {path}")
