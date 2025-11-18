# qtrader

A minimal quantitative trading backtesting framework with:
- CSV data loading
- SMA/EMA indicators
- SMA crossover strategy example
- Paper broker with slippage/fees
- Backtesting engine and metrics

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e . 
python -m qtrader --help
```

## Bonus: Macaroon Tetris (Web)
Looking for a break from backtests? A sweet, pastel-themed Tetris clone lives in `web/macaroon-tetris/`.

1. Launch any static server (for example: `cd web/macaroon-tetris && python -m http.server 8000`).
2. Open `http://localhost:8000` in your browser.
3. Hit `Start` (or press `Enter`) and enjoy the macaroon-colored gameplay.

Extras:
- Hold queue (Shift/C) with dedicated preview canvas.
- Lightweight Web Audio chimes for moves, drops, line clears, level-ups, and game over.
- Built-in 300×250 + 728×90 ad placeholders so sponsor creatives can drop right in.
