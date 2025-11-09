# scripts/enrich_books.py
# Enrich a markets DataFrame (from adapters/poly_gamma.df) with real CLOB books.
# Adds columns prefixed with book_* WITHOUT mutating the original df.

import json
import numpy as np
import pandas as pd
from typing import List

from adapters.clob_books import fetch_books, top_of_book

def _parse_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s)
            except Exception:
                return None
    return None

def add_books_signals(df: pd.DataFrame, max_rows: int = 200, fee_buffer: float = 0.003) -> pd.DataFrame:
    """
    Returns a NEW DataFrame with columns:
      - book_bids (list[float]), book_asks (list[float])
      - book_bid_sum, book_ask_sum
      - book_bid_size_sum, book_ask_size_sum
      - book_underround (ask_sum < 1 - fee_buffer)
      - book_overround  (bid_sum > 1 + fee_buffer)
    Only uses rows (up to max_rows) that have clobTokenIds; others remain NaN.
    """
    out = df.copy(deep=False)  # don't mutate original

    # rows we’ll actually query (cap to save cost)
    if "clobTokenIds" not in out.columns:
        # nothing to enrich
        for col in ["book_bids","book_asks","book_bid_sum","book_ask_sum",
                    "book_bid_size_sum","book_ask_size_sum","book_underround","book_overround"]:
            out[col] = np.nan
        return out

    idx_with_tokens = []
    tokens_per_row: List[List[str]] = []

    for idx, cell in out["clobTokenIds"].items():
        toks = _parse_list(cell)
        if toks:
            idx_with_tokens.append(idx)
            tokens_per_row.append(toks)
            if len(idx_with_tokens) >= max_rows:
                break

    # Prepare storage
    book_bids = pd.Series([np.nan]*len(out), index=out.index, dtype="object")
    book_asks = pd.Series([np.nan]*len(out), index=out.index, dtype="object")
    bid_sum   = pd.Series(np.nan, index=out.index, dtype="float64")
    ask_sum   = pd.Series(np.nan, index=out.index, dtype="float64")
    bid_size_sum = pd.Series(np.nan, index=out.index, dtype="float64")
    ask_size_sum = pd.Series(np.nan, index=out.index, dtype="float64")
    underround = pd.Series(np.nan, index=out.index, dtype="float64")
    overround  = pd.Series(np.nan, index=out.index, dtype="float64")

    # Fetch books row-by-row (simplest to test; can batch later)
    for idx, toks in zip(idx_with_tokens, tokens_per_row):
        try:
            books = fetch_books(toks)
            bids, asks = [], []
            b_sizes, a_sizes = [], []
            for b in books:
                bb, ba, bs, aS = top_of_book(b)
                bids.append(bb); asks.append(ba)
                b_sizes.append(bs); a_sizes.append(aS)

            # sums ignoring None
            bsum = float(sum(x for x in bids if isinstance(x, (int, float))))
            asum = float(sum(x for x in asks if isinstance(x, (int, float))))
            bss  = float(sum(x for x in b_sizes if isinstance(x, (int, float))))
            ass  = float(sum(x for x in a_sizes if isinstance(x, (int, float))))

            book_bids.at[idx] = bids
            book_asks.at[idx] = asks
            bid_sum.at[idx] = bsum
            ask_sum.at[idx] = asum
            bid_size_sum.at[idx] = bss
            ask_size_sum.at[idx] = ass
            underround.at[idx] = 1.0 if (asum < (1.0 - fee_buffer)) else 0.0
            overround.at[idx]  = 1.0 if (bsum > (1.0 + fee_buffer)) else 0.0

        except Exception:
            # leave NaNs if request failed
            continue

    # Attach new columns (prefixed)
    out = out.assign(
        book_bids=book_bids,
        book_asks=book_asks,
        book_bid_sum=bid_sum,
        book_ask_sum=ask_sum,
        book_bid_size_sum=bid_size_sum,
        book_ask_size_sum=ask_size_sum,
        book_underround=underround,
        book_overround=overround,
    )
    return out

if __name__ == "__main__":
    # Quick test runner: enrich the adapter df and print a small view
    import sys, pathlib
    ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    try:
        from adapters.poly_gamma import df as base_df
    except Exception as e:
        print("[enrich_books] Could not import adapters.poly_gamma.df:", e)
        sys.exit(1)

    # Filter to active + liquidity present to keep calls focused
    liq = None
    for c in ("liquidityNum","liquidityClob","liquidity"):
        if c in base_df.columns:
            liq = pd.to_numeric(base_df[c], errors="coerce") if liq is None else liq.fillna(pd.to_numeric(base_df[c], errors="coerce"))
    mask = (base_df.get("active", True) == True)
    if liq is not None:
        mask &= liq.notna()

    df_small = base_df[mask].copy()
    print(f"[enrich_books] Input rows (active/liquid): {len(df_small)} / {len(base_df)}")

    enriched = add_books_signals(df_small, max_rows=50, fee_buffer=0.003)
    print(f"[enrich_books] Enriched rows: {len(enriched)}")

    # Show candidates with gross under/overround
    view_cols = [c for c in [
        "id","marketId","question","slug",
        "outcomes","outcomePrices","clobTokenIds",
        "book_bid_sum","book_ask_sum","book_underround","book_overround",
        "book_bid_size_sum","book_ask_size_sum"
    ] if c in enriched.columns]

    cands = enriched[(enriched["book_underround"] == 1.0) | (enriched["book_overround"] == 1.0)]
    print(f"[enrich_books] Arbitrage flags: {len(cands)}")
    print(cands[view_cols].head(15).to_string(index=False))
