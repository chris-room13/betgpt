import requests, json, pandas as pd

# -------- 1) Fetch --------
r = requests.get("https://gamma-api.polymarket.com/events?closed=false", timeout=15)
response = r.json()

# -------- 2) Build DF (keep all original market fields; add event_ fields) --------
def markets_df_from_events_response(response, parse_helpers=True):
    if isinstance(response, list):
        events = response
    elif isinstance(response, dict):
        events = response.get("events") or response.get("data") or [response]
        if not isinstance(events, list):
            events = [events]
    else:
        raise TypeError("Unsupported response type")

    rows = []
    for ev in events:
        markets = ev.get("markets") or ev.get("marketsData") or []
        ev_meta = {f"event_{k}": v for k, v in ev.items() if k not in ("markets", "marketsData")}
        for m in markets:
            row = dict(m)
            row.update(ev_meta)
            rows.append(row)

    df = pd.DataFrame(rows)

    if parse_helpers and not df.empty:
        # parse JSON-in-strings into helper columns (raw kept)
        for col in ["outcomes", "shortOutcomes", "outcomePrices", "clobTokenIds", "umaResolutionStatuses"]:
            if col in df.columns:
                df[col + "_list"] = df[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith("[") else x
                )
        # datetimes (raw kept)
        for col in ["startDate","endDate","createdAt","updatedAt","acceptingOrdersTimestamp",
                    "event_startDate","event_endDate","event_createdAt","event_updatedAt"]:
            if col in df.columns:
                df[col + "_dt"] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df

df = markets_df_from_events_response(response, parse_helpers=True)

# -------- 3) Filter: only active markets --------
if "active" in df.columns:
    df = df[df["active"] == True]

# -------- 4) Liquidity: make numeric and drop NaNs --------
# Coalesce common fields into one numeric column `liquidity_num`
import numpy as np

# init
df["liquidity_num"] = np.nan  # dtype float64 + NaN

# coalesce sources into one numeric column
for c in ["liquidityNum", "liquidityClob", "liquidity"]:
    if c in df.columns:
        df["liquidity_num"] = df["liquidity_num"].fillna(pd.to_numeric(df[c], errors="coerce"))

# drop NaNs
df = df.dropna(subset=["liquidity_num"])



if __name__ == "__main__":
    print("Fetching + building + filtering...")
    print(df.shape)   # df comes from the script you pasted
    print(df.head(5))
