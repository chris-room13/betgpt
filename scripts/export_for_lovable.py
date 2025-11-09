# scripts/export_for_lovable.py
import json, pandas as pd, numpy as np
from pathlib import Path

from adapters.poly_gamma import df as base_df
from scripts.enrich_books import add_books_signals  # remove if you don't want book_* fields

# 1) build the DF for the app
enriched = add_books_signals(base_df, max_rows=400, fee_buffer=0.003)

# 2) add Polymarket URL
if "slug" in enriched.columns:
    enriched["url"] = "https://polymarket.com/event/" + enriched["slug"].astype(str)
else:
    enriched["url"] = None

# 3) choose columns
cols = [c for c in [
    "id","marketId","question","slug","url","endDate","active","acceptingOrders",
    "outcomes","outcomePrices","bestBid","bestAsk",
    "liquidityNum","liquidityClob","liquidity","volumeNum",
    "book_bids","book_asks","book_bid_sum","book_ask_sum",
    "book_underround","book_overround","book_bid_size_sum","book_ask_size_sum"
] if c in enriched.columns]
view = enriched[cols].copy()

# 4) JSON-friendly records
records = json.loads(view.replace({np.nan: None}).to_json(orient="records"))

# 5) write to <project_root>/public/markets.json (create folder if needed)
ROOT = Path(__file__).resolve().parents[1]
out_path = ROOT / "public" / "markets.json"
out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, separators=(",", ":"))

print(f"Wrote {out_path} with {len(records)} rows including URLs")
