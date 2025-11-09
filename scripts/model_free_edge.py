# scripts/model_free_edge.py
# Finds model-free edges:
# - uses adapters/poly_gamma.df (base)
# - enriches with CLOB books (book_* columns)
# - computes naive anomalies (no external calls beyond /books)
# - does NOT mutate the base df

import sys, pathlib, json
import numpy as np
import pandas as pd

# -------- path shim so "adapters" & "scripts" are importable when run directly --------
ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ======== 1) IMPORT BASE + ENRICH WITH BOOKS ========
try:
    from adapters.poly_gamma import df as base_df
except Exception as e:
    print("[model_free_edge] ERROR: couldn't import adapters.poly_gamma.df:", e, file=sys.stderr)
    print("Run from project root or set PYTHONPATH=.", file=sys.stderr)
    sys.exit(1)

try:
    from scripts.enrich_books import add_books_signals
except Exception as e:
    print("[model_free_edge] ERROR: couldn't import scripts.enrich_books.add_books_signals:", e, file=sys.stderr)
    sys.exit(1)

# keep this small while testing; increase later
FEE_BUFFER = 0.003
enriched = add_books_signals(base_df, max_rows=200, fee_buffer=FEE_BUFFER)

# ======== 2) NAIVE ANOMALIES (no mutation of enriched) ========
def _parse_list(series: pd.Series) -> pd.Series:
    def parse_cell(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    return json.loads(s)
                except Exception:
                    return np.nan
        return np.nan
    return series.apply(parse_cell)

def compute_edges(df: pd.DataFrame) -> pd.DataFrame:
    # Parsed lists (derived only)
    outcomes_col = df["outcomes"] if "outcomes" in df.columns else pd.Series(np.nan, index=df.index)
    prices_col   = df["outcomePrices"] if "outcomePrices" in df.columns else pd.Series(np.nan, index=df.index)

    outcomes_list = _parse_list(outcomes_col)
    outcomePrices_list = _parse_list(prices_col).apply(
        lambda xs: [float(x) for x in xs] if isinstance(xs, list) else np.nan
    )

    # Convenience: price_yes / price_no if present
    def _pick(series_prices, series_outcomes, name):
        out = []
        for outs, prs in zip(series_outcomes, series_prices):
            if not isinstance(outs, list) or not isinstance(prs, list):
                out.append(np.nan); continue
            try:
                i = outs.index(name)
                out.append(float(prs[i]) if i < len(prs) else np.nan)
            except ValueError:
                out.append(np.nan)
        return pd.Series(out, index=series_prices.index, dtype="float64")

    price_yes = _pick(outcomePrices_list, outcomes_list, "Yes")
    price_no  = _pick(outcomePrices_list, outcomes_list, "No")

    # Sum sanity & degeneracy
    ask_sum = outcomePrices_list.apply(lambda xs: float(sum(xs)) if isinstance(xs, list) and len(xs) >= 2 else np.nan)
    sum_off = ask_sum.sub(1.0).abs()
    sum_bad = sum_off > 0.002
    EPS = 1e-6
    def _degenerate(xs):
        if not isinstance(xs, list) or len(xs) != 2:
            return False
        a, b = float(xs[0]), float(xs[1])
        return (abs(a-0) <= EPS and abs(b-1) <= EPS) or (abs(a-1) <= EPS and abs(b-0) <= EPS)
    degenerate_prices = outcomePrices_list.apply(_degenerate)

    # Spread/mid for single exposed outcome (if present)
    bestBid_num = pd.to_numeric(df["bestBid"], errors="coerce") if "bestBid" in df.columns else pd.Series(np.nan, index=df.index)
    bestAsk_num = pd.to_numeric(df["bestAsk"], errors="coerce") if "bestAsk" in df.columns else pd.Series(np.nan, index=df.index)
    mid = (bestBid_num + bestAsk_num) / 2
    rel_spread = (bestAsk_num - bestBid_num) / mid
    crossed_or_zero = bestBid_num.notna() & bestAsk_num.notna() & (bestBid_num >= bestAsk_num - 1e-6)

    # Stale YES ask vs bestAsk (only if both exist)
    stale_yes_ask = price_yes.notna() & bestAsk_num.notna() & (price_yes.sub(bestAsk_num).abs() > 0.003)

    # Tick alignment
    tick = pd.to_numeric(df["orderPriceMinTickSize"], errors="coerce") if "orderPriceMinTickSize" in df.columns else pd.Series(np.nan, index=df.index)
    def _on_tick(price, t):
        if pd.isna(price) or pd.isna(t) or t <= 0: return True
        return abs((price / t) - round(price / t)) <= 1e-9
    yes_on_tick = pd.Series([_on_tick(p, t) for p, t in zip(price_yes, tick)], index=df.index)
    no_on_tick  = pd.Series([_on_tick(p, t) for p, t in zip(price_no,  tick)], index=df.index)

    # Liquidity coalesce (numeric) without changing df
    liq_num = pd.Series(np.nan, index=df.index, dtype="float64")
    for c in ("liquidityNum", "liquidityClob", "liquidity"):
        if c in df.columns:
            liq_num = liq_num.fillna(pd.to_numeric(df[c], errors="coerce"))

    keep_cols = [c for c in [
        "id","marketId","question","slug","endDate","outcomes","outcomePrices",
        "active","acceptingOrders","volumeNum","liquidityNum","liquidityClob","liquidity",
        # from books (if any)
        "book_bids","book_asks","book_bid_sum","book_ask_sum","book_bid_size_sum","book_ask_size_sum",
        "book_underround","book_overround"
    ] if c in df.columns]

    base = df[keep_cols].copy()

    edges = pd.concat([
        base,
        pd.DataFrame({
            "outcomes_list": outcomes_list,
            "outcomePrices_list": outcomePrices_list,
            "price_yes": price_yes,
            "price_no":  price_no,
            "ask_sum": ask_sum,
            "sum_off": sum_off,
            "sum_bad": sum_bad,
            "degenerate_prices": degenerate_prices,
            "bestBid_num": bestBid_num,
            "bestAsk_num": bestAsk_num,
            "mid": mid,
            "rel_spread": rel_spread,
            "crossed_or_zero": crossed_or_zero,
            "stale_yes_ask": stale_yes_ask,
            "yes_on_tick": yes_on_tick,
            "no_on_tick":  no_on_tick,
            "liquidity_num": liq_num,
        }, index=df.index)
    ], axis=1)

    return edges

# Build edges on the enriched df (has book_* columns)
edges = compute_edges(enriched)

# ======== 3) DISPLAY / OUTPUT ========
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

def get(col, default=None):
    return edges[col] if col in edges.columns else default

# Quality gates (purely in-DF)
liq_series = get("liquidity_num", pd.Series([np.nan]*len(edges)))
quality = (
    (get("active", True) == True) &
    (get("acceptingOrders", True) == True) &
    (liq_series.notna())
)

# Edge (books-based, real arbitrage tests)
edge_mask = (
    (get("book_underround", pd.Series([0]*len(edges))) == 1.0) |
    (get("book_overround",  pd.Series([0]*len(edges))) == 1.0)
)

# Optional: require some top-of-book size
size_ok = (
    get("book_ask_size_sum", pd.Series([0]*len(edges))).fillna(0) >= 1
) | (
    get("book_bid_size_sum", pd.Series([0]*len(edges))).fillna(0) >= 1
)

candidates_mask = quality & edge_mask & size_ok

view_cols = [c for c in [
    "id","marketId","question","slug","endDate",
    "book_bids","book_asks","book_bid_sum","book_ask_sum",
    "book_bid_size_sum","book_ask_size_sum",
    "book_underround","book_overround",
    "price_yes","price_no","bestBid_num","bestAsk_num","rel_spread",
    "liquidity_num","volumeNum","acceptingOrders"
] if c in edges.columns]

candidates = edges.loc[candidates_mask, view_cols].copy()

# Sort: underround first, then biggest (bid_sum - ask_sum), then more size
if {"book_bid_sum","book_ask_sum"} <= set(candidates.columns):
    candidates["gross_gap"] = candidates["book_bid_sum"].fillna(0) - candidates["book_ask_sum"].fillna(0)
    candidates = candidates.sort_values(
        by=["book_underround","gross_gap","book_bid_size_sum","book_ask_size_sum"],
        ascending=[False, False, False, False]
    )

# always print something useful
print(f"[model_free_edge] rows={len(edges)}, cols={edges.shape[1]}")
print(f"[model_free_edge] edge candidates={len(candidates)} (quality & books-based)")

if len(candidates) > 0:
    print(candidates.head(30).to_string(index=False))
else:
    # fallback: show top liquid rows with book sums to verify plumbing
    fallback_cols = [c for c in [
        "id","marketId","question","slug","endDate",
        "book_bid_sum","book_ask_sum","book_bid_size_sum","book_ask_size_sum",
        "liquidity_num","acceptingOrders"
    ] if c in edges.columns]
    fallback = edges.loc[quality, fallback_cols].copy()
    if "liquidity_num" in fallback.columns:
        fallback = fallback.sort_values("liquidity_num", ascending=False)
    print("[model_free_edge] No candidates; showing top by liquidity:")
    print(fallback.head(20).to_string(index=False))

# Artifacts
try:
    edges.to_csv("edges_snapshot.csv", index=False)
    candidates.to_csv("edge_candidates_books.csv", index=False)
    print("[model_free_edge] wrote edges_snapshot.csv and edge_candidates_books.csv")
except Exception:
    pass
