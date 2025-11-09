# BetGPT — Hackathon Prototype

This repo contains our hackathon project **BetGPT**, a pipeline that pulls markets from Polymarket (Gamma + order books), cleans/enriches them in Python, and exports a JSON used by a simple UI (Lovable).

## Why this repo exists
We’re beginners and built this fast for a submission. It’s **not fully reproducible** or production-ready:
- Hard-coded bits, minimal error handling
- No guaranteed environment setup
- APIs can change; results may vary

Still, we wanted to **publish** what we have for transparency and as a learning milestone.

## What it (roughly) does
- Fetch Polymarket markets and order books
- Parse/clean into a DataFrame
- Compute model-free edge hints (e.g., under/overround from top-of-book)
- Export to `public/markets.json` (includes a link back to the market)

## Status / limitations
- Prototype quality; expect rough edges
- No full docs or tests
- Use at your own risk
