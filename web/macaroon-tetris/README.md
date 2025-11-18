# Macaroon Tetris

A lightweight, dependency-free Tetris implementation rendered on HTML5 canvas with a soft "macaroon" visual theme.

## Design Notes
- **Palette**: Soft candy-inspired pastels (`#f7c5d1`, `#ffd6a5`, `#fef7c3`, `#cdeac0`, `#cddafd`, `#ffcfe1`, `#bde0fe`).
- **Layout**: Main game board canvas (10×20 grid) centered with rounded container, next-piece preview, scoreboard, and helper instructions stacked on the side for desktops and below for narrow screens.
- **Typography**: Rounded, friendly font stack (`"Poppins", "Avenir", "Segoe UI", sans-serif`).
- **Controls**: Arrow keys (move/soft drop), `ArrowUp` (rotate), `Space` (hard drop), `P` (pause/resume), `Enter` (restart).
- **Game Loop**: RequestAnimationFrame-driven loop with adjustable gravity interval; board logic implemented in vanilla ES modules.
- **State**: Board matrix (10×20), active piece, queue with next preview, scoring (Tetris guideline scoring with level-based speedup), pause flag.
- **Responsiveness**: CSS grid + flexbox to adapt from desktop to mobile; canvases scale with CSS while drawing uses logical pixels for crispness.

## Files
- `index.html` – Static markup for the game shell.
- `styles.css` – Pastel macaroon theme and responsive layout.
- `game.js` – Tetris logic, rendering, and controls.

## Getting Started
Run the files through any static HTTP server (local files may be blocked by browsers due to module scripts):

```bash
cd web/macaroon-tetris
python -m http.server 8000
```

Then open `http://localhost:8000` and press **Start** (or hit `Enter`).

## Gameplay Highlights
- Standard 10×20 Tetris well with bag randomizer and next-piece preview.
- Scoring rewards multi-line clears; level/speed increase every 10 cleared lines.
- Controls: `←/→` move, `↑` rotate, `↓` soft drop, `Space` hard drop, `P` pause/resume, `Enter` restart.
