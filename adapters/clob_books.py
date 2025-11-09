# adapters/clob_books.py
import requests
from typing import List, Tuple, Optional

CLOB_BOOKS_URL = "https://clob.polymarket.com/books"

def fetch_books(token_ids: List[str], timeout: float = 8.0) -> list:
    """
    Batch-fetch order books for a list of outcome token_ids.
    Returns a list of book dicts (same order as token_ids).
    """
    if not token_ids:
        return []
    payload = [{"token_id": t} for t in token_ids]
    r = requests.post(CLOB_BOOKS_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # Expected: list same length as token_ids
    return data

def top_of_book(book: dict) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Extract top-of-book (best bid/ask + sizes) from a single book dict.
    Returns (best_bid, best_ask, bid_size, ask_size) as floats or None.
    """
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    bb = float(bids[0]["price"]) if bids else None
    ba = float(asks[0]["price"]) if asks else None
    bs = float(bids[0]["size"])  if bids else None
    aS = float(asks[0]["size"])  if asks else None
    return bb, ba, bs, aS
