import camelot
import pandas as pd

# 1) Path to your uploaded PDF:
PDF_PATH = "/mnt/data/NFLDK2025_CS_ClayProjections2025.pdf"

# 2) Teams in the order ESPN prints them (pages 2–33 in the PDF):
TEAM_ORDER = [
    "Arizona Cardinals","Atlanta Falcons","Baltimore Ravens","Buffalo Bills",
    "Carolina Panthers","Chicago Bears","Cincinnati Bengals","Cleveland Browns",
    "Dallas Cowboys","Denver Broncos","Detroit Lions","Green Bay Packers",
    "Houston Texans","Indianapolis Colts","Jacksonville Jaguars","Kansas City Chiefs",
    "Las Vegas Raiders","Los Angeles Chargers","Los Angeles Rams","Miami Dolphins",
    "Minnesota Vikings","New England Patriots","New Orleans Saints","New York Giants",
    "New York Jets","Philadelphia Eagles","Pittsburgh Steelers","San Francisco 49ers",
    "Seattle Seahawks","Tampa Bay Buccaneers","Tennessee Titans","Washington Commanders"
]

# 3) Abbreviation → full‑name mapping for opponents:
ABBR_MAP = {
    'ARI':'Arizona Cardinals','ATL':'Atlanta Falcons','BAL':'Baltimore Ravens',
    'BUF':'Buffalo Bills','CAR':'Carolina Panthers','CHI':'Chicago Bears',
    'CIN':'Cincinnati Bengals','CLE':'Cleveland Browns','DAL':'Dallas Cowboys',
    'DEN':'Denver Broncos','DET':'Detroit Lions','GB':'Green Bay Packers',
    'HOU':'Houston Texans','IND':'Indianapolis Colts','JAX':'Jacksonville Jaguars',
    'KC':'Kansas City Chiefs','LV':'Las Vegas Raiders','LAC':'Los Angeles Chargers',
    'LAR':'Los Angeles Rams','MIA':'Miami Dolphins','MIN':'Minnesota Vikings',
    'NE':'New England Patriots','NO':'New Orleans Saints','NYG':'New York Giants',
    'NYJ':'New York Jets','PHI':'Philadelphia Eagles','PIT':'Pittsburgh Steelers',
    'SF':'San Francisco 49ers','SEA':'Seattle Seahawks','TB':'Tampa Bay Buccaneers',
    'TEN':'Tennessee Titans','WAS':'Washington Commanders'
}

all_games = []

# 4) Loop over pages 2–33 (printed), i.e. camelot pages="2-33"
for page_num, team in zip(range(2, 34), TEAM_ORDER):
    tables = camelot.read_pdf(PDF_PATH, pages=str(page_num), flavor='stream')
    if not tables:
        continue
    df = tables[0].df

    # find the row that contains “Wk” & “Opp” (that’s the weekly‑table header)
    hdr_idx = next(
        (i for i,row in df.iterrows() if 'Wk' in row.values and 'Opp' in row.values),
        None
    )
    if hdr_idx is None:
        continue

    header = df.iloc[hdr_idx].tolist()
    data  = df.iloc[hdr_idx+1 : hdr_idx+19]   # next 18 rows → Weeks 1–18

    # locate the column‑indexes:
    wk_i      = header.index('Wk')
    opp_idxs  = [i for i,v in enumerate(header) if v == 'Opp']
    opp_tm_i  = opp_idxs[0]
    opp_pr_i  = opp_idxs[1]
    loc_i     = header.index('Loc')
    tmproj_i  = header.index('Tm')
    winp_i    = header.index('Win Prob')

    # pull out each week's row
    for _, row in data.iterrows():
        w = row[wk_i].strip()
        if not w.isdigit():
            continue
        wk = int(w)
        opp_abbr = row[opp_tm_i].strip()
        loc      = row[loc_i].strip()
        tproj    = float(row[tmproj_i])
        oproj    = float(row[opp_pr_i])
        wpct     = row[winp_i].strip()

        opp_full = ABBR_MAP.get(opp_abbr, opp_abbr)
        all_games.append({
            'Week': wk,
            'Team': team,
            'Opp': opp_full,
            'Loc': loc,
            'TeamProj': tproj,
            'OppProj': oproj,
            'WinPct': wpct,
            'Total': tproj + oproj
        })

# 5) De‑duplicate (each game shows up twice – once on each team's page)
df = pd.DataFrame(all_games)
df['Key'] = df.apply(lambda r: tuple(sorted([r['Team'], r['Opp']])), axis=1)
df_unique = df.drop_duplicates(subset=['Week','Key'])

# 6) For each week, grab the top 3 by implied total
top3 = (
    df_unique
      .sort_values(['Week','Total'], ascending=[True,False])
      .groupby('Week')
      .head(3)
      .reset_index(drop=True)
)

# 7) Print it out
for wk in range(1,19):
    block = top3[top3['Week']==wk]
    print(f"\nWeek {wk:2d}:")
    for _, r in block.iterrows():
        print(f"  • {r['Team']} @ {r['Opp']}  ⇒  {r['Total']:.1f} pts (proj {r['TeamProj']:.1f}–{r['OppProj']:.1f}, {r['WinPct']})")
