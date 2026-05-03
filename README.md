# quoter

Venue-agnostic market-making strategy framework — by [kcolbchain](https://kcolbchain.com) (est. 2015).

The strategy core. Pluggable quoting strategies, a backtest engine, and exchange connectors. Bring your own venue.

> Looking for the **on-chain RWA stack** (vault contracts, oracle adapter, EVM execution, risk manager)? See [`kcolbchain/meridian`](https://github.com/kcolbchain/meridian) — meridian builds on quoter's strategy core and adds chain connectivity, on-chain execution, and ERC-4626 LP vaults.

## What's in the box

- **Strategies** — `constant_spread`, `adaptive_spread` (volatility + inventory aware). Plug in your own by subclassing `Strategy`.
- **Backtest engine** — tick-driven, deterministic. CSV/Parquet export of fills.
- **Connectors** — base interface + reference connectors. Drop in CEX WebSockets / DEX RPC / your own venue.
- **Oracle abstraction** — mock + real `PriceFeed` source. Same interface for tests and live.
- **Inventory + risk hooks** — strategies see current position so they can skew quotes against directional exposure.

## Quick start

```bash
git clone https://github.com/kcolbchain/quoter.git
cd quoter
pip install -r requirements.txt

# Backtest
python run.py --simulate --pair ETH/USDC --spread 0.5 --ticks 200

# Backtest with adaptive strategy
python run.py --simulate --config config/default.yaml --output fills

# Live mode (requires venue connector + keys; stub by default)
python run.py --config config/live.yaml
```

## Strategies

| Strategy | What it does |
|---|---|
| `constant_spread` | Fixed bid/ask spread around mid — baseline. |
| `adaptive_spread` | Spread widens with realised volatility, narrows in calm markets, skews against inventory. |

Add your own by subclassing `Strategy` in `src/strategies/`. `adaptive_spread.py` is a clean reference shape.

## Architecture

```
┌─────────────────────────────────────────┐
│           Strategy (pluggable)          │
│   constant / adaptive / your own        │
├─────────────────────────────────────────┤
│  Agent loop  ─►  Oracle  ─►  Connector  │
│       │              │            │     │
│       └─── Inventory + risk hooks ──────┤
├─────────────────────────────────────────┤
│             Backtest engine             │
└─────────────────────────────────────────┘
```

## Project structure

```
src/
  agents/        — base_agent, rwa_market_maker (reference agent)
  strategies/    — constant_spread, adaptive_spread, your own
  oracle/        — PriceFeed abstraction (mock + live)
  connectors/    — venue interface; bring your own CEX/DEX adapter
  backtest/      — tick-driven engine + fill export
  utils/         — config, logging, inventory helpers
config/          — default.yaml, live.yaml
tests/           — pytest suite
```

## Where this fits

- **Strategy library you can drop into any venue** — quoter is on its own here. The strategies and backtest engine don't assume a chain or a centralised venue.
- **Reference on-chain deployment** — see [`kcolbchain/meridian`](https://github.com/kcolbchain/meridian). Meridian uses quoter's strategies and ships ERC-4626 vault contracts, an oracle adapter, EVM execution, and a risk manager on top.
- **Stablecoin / RWA issuance** — quoter doesn't issue tokens. See [`kcolbchain/stablecoin-toolkit`](https://github.com/kcolbchain/stablecoin-toolkit).
- **Audit hardening before a strategy goes live** — see [`kcolbchain/audit-checklist`](https://github.com/kcolbchain/audit-checklist).

## Running the tests

```bash
pytest -q
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CONTRIBUTORS.md](CONTRIBUTORS.md). Issues tagged `good-first-issue` are great entry points.

## Working with kcolbchain

We build, deploy, and run market-making infrastructure for partner protocols. If you'd like to talk about a paid integration or managed market-making, see [kcolbchain.com/work-with-us](https://kcolbchain.com/work-with-us/).

## Links

- **Docs:** https://docs.kcolbchain.com/quoter/
- **All projects:** https://docs.kcolbchain.com/
- **kcolbchain:** https://kcolbchain.com

## License

MIT — see [LICENSE](LICENSE)

---

*Founded by [Abhishek Krishna](https://abhishekkrishna.com) • GitHub: [@abhicris](https://github.com/abhicris)*
