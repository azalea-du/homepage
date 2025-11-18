const BOARD_COLS = 10;
const BOARD_ROWS = 20;
const BASE_DROP_INTERVAL = 900;
const MIN_DROP_INTERVAL = 120;

const TETROMINOES = {
  I: {
    color: "#bde0fe",
    rotations: [
      [
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ],
      [
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
      ],
    ],
  },
  J: {
    color: "#cddafd",
    rotations: [
      [
        [1, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
      ],
      [
        [0, 1, 1],
        [0, 1, 0],
        [0, 1, 0],
      ],
      [
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 1],
      ],
      [
        [0, 1, 0],
        [0, 1, 0],
        [1, 1, 0],
      ],
    ],
  },
  L: {
    color: "#ffd6a5",
    rotations: [
      [
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 0],
      ],
      [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
      ],
      [
        [0, 0, 0],
        [1, 1, 1],
        [1, 0, 0],
      ],
      [
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
      ],
    ],
  },
  O: {
    color: "#fef7c3",
    rotations: [
      [
        [1, 1],
        [1, 1],
      ],
    ],
  },
  S: {
    color: "#cdeac0",
    rotations: [
      [
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 0],
      ],
      [
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
      ],
    ],
  },
  T: {
    color: "#f7c5d1",
    rotations: [
      [
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
      ],
      [
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 0],
      ],
      [
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0],
      ],
      [
        [0, 1, 0],
        [1, 1, 0],
        [0, 1, 0],
      ],
    ],
  },
  Z: {
    color: "#ffcfe1",
    rotations: [
      [
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 0],
      ],
      [
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
      ],
    ],
  },
};

const lineRewards = [0, 100, 300, 500, 800];

const boardCanvas = document.getElementById("board");
const nextCanvas = document.getElementById("next");
const boardCtx = boardCanvas.getContext("2d");
const nextCtx = nextCanvas.getContext("2d");

const scoreEl = document.getElementById("score");
const linesEl = document.getElementById("lines");
const levelEl = document.getElementById("level");
const statusPill = document.getElementById("status-pill");
const toggleBtn = document.getElementById("toggle-btn");

boardCtx.imageSmoothingEnabled = false;
nextCtx.imageSmoothingEnabled = false;

const cellSize = Math.floor(boardCanvas.width / BOARD_COLS);

const state = {
  board: createMatrix(BOARD_COLS, BOARD_ROWS),
  piece: null,
  nextPiece: null,
  bag: [],
  score: 0,
  lines: 0,
  level: 1,
  dropInterval: BASE_DROP_INTERVAL,
  dropCounter: 0,
  lastTime: 0,
  animationId: null,
  running: false,
  paused: false,
};

function createMatrix(width, height) {
  return Array.from({ length: height }, () => Array(width).fill(null));
}

function shuffle(array) {
  for (let i = array.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

function refillBag() {
  state.bag = shuffle(Object.keys(TETROMINOES));
}

function takePiece() {
  if (!state.bag.length) {
    refillBag();
  }
  const key = state.bag.pop();
  const blueprint = TETROMINOES[key];
  return {
    key,
    rotationIndex: 0,
    matrix: blueprint.rotations[0],
    color: blueprint.color,
    position: { x: Math.floor(BOARD_COLS / 2) - 2, y: -1 },
  };
}

function spawnPiece() {
  state.piece = state.nextPiece || takePiece();
  state.nextPiece = takePiece();
  state.piece.position = { x: Math.floor(BOARD_COLS / 2) - 2, y: -1 };
  if (collide(state.board, state.piece)) {
    return gameOver();
  }
  drawNext();
}

function startGame() {
  state.board = createMatrix(BOARD_COLS, BOARD_ROWS);
  state.score = 0;
  state.lines = 0;
  state.level = 1;
  state.dropInterval = BASE_DROP_INTERVAL;
  state.dropCounter = 0;
  state.lastTime = 0;
  state.running = true;
  state.paused = false;
  if (state.animationId) {
    cancelAnimationFrame(state.animationId);
  }
  refillBag();
  state.nextPiece = takePiece();
  spawnPiece();
  updateStats();
  updateStatus("running");
  toggleBtn.textContent = "Pause";
  setButtonPressed(true);
  state.animationId = requestAnimationFrame(update);
}

function pauseGame() {
  if (!state.running || state.paused) return;
  state.paused = true;
  cancelAnimationFrame(state.animationId);
  state.animationId = null;
  updateStatus("paused");
  toggleBtn.textContent = "Resume";
  setButtonPressed(false);
}

function resumeGame() {
  if (!state.running || !state.paused) return;
  state.paused = false;
  state.lastTime = performance.now();
  updateStatus("running");
  toggleBtn.textContent = "Pause";
  setButtonPressed(true);
  state.animationId = requestAnimationFrame(update);
}

function gameOver() {
  state.running = false;
  state.paused = false;
  cancelAnimationFrame(state.animationId);
  state.animationId = null;
  updateStatus("idle", "topped out");
  toggleBtn.textContent = "Restart";
  setButtonPressed(false);
  draw();
}

function update(time = 0) {
  if (!state.running || state.paused) return;
  const delta = time - state.lastTime;
  state.lastTime = time;
  state.dropCounter += delta;
  if (state.dropCounter > state.dropInterval) {
    drop();
  }
  draw();
  state.animationId = requestAnimationFrame(update);
}

function drop() {
  state.dropCounter = 0;
  if (!movePiece(0, 1)) {
    lockPiece();
  }
}

function hardDrop() {
  if (!state.piece) return;
  let distance = 0;
  while (movePiece(0, 1)) {
    distance += 1;
  }
  state.score += distance * 2;
  lockPiece();
}

function movePiece(offsetX, offsetY) {
  if (!state.piece) return false;
  state.piece.position.x += offsetX;
  state.piece.position.y += offsetY;
  if (collide(state.board, state.piece)) {
    state.piece.position.x -= offsetX;
    state.piece.position.y -= offsetY;
    return false;
  }
  return true;
}

function rotatePiece(dir = 1) {
  if (!state.piece) return;
  const blueprint = TETROMINOES[state.piece.key];
  const len = blueprint.rotations.length;
  const prevIndex = state.piece.rotationIndex;
  const nextIndex = (prevIndex + dir + len) % len;
  const nextMatrix = blueprint.rotations[nextIndex];

  const originalX = state.piece.position.x;
  state.piece.rotationIndex = nextIndex;
  state.piece.matrix = nextMatrix;

  const offsets = [0, -1, 1, -2, 2];
  for (const offset of offsets) {
    state.piece.position.x = originalX + offset;
    if (!collide(state.board, state.piece)) {
      return;
    }
  }

  state.piece.rotationIndex = prevIndex;
  state.piece.matrix = blueprint.rotations[prevIndex];
  state.piece.position.x = originalX;
}

function collide(board, piece) {
  const { matrix, position } = piece;
  for (let y = 0; y < matrix.length; y += 1) {
    for (let x = 0; x < matrix[y].length; x += 1) {
      if (!matrix[y][x]) continue;
      const boardX = x + position.x;
      const boardY = y + position.y;
      if (boardX < 0 || boardX >= BOARD_COLS || boardY >= BOARD_ROWS) {
        return true;
      }
      if (boardY >= 0 && board[boardY][boardX]) {
        return true;
      }
    }
  }
  return false;
}

function lockPiece() {
  const { matrix, position, color } = state.piece;
  matrix.forEach((row, y) => {
    row.forEach((value, x) => {
      if (!value) return;
      const boardY = y + position.y;
      if (boardY < 0) return;
      const boardX = x + position.x;
      state.board[boardY][boardX] = color;
    });
  });
  const cleared = sweepLines();
  if (cleared > 0) {
    state.score += lineRewards[cleared] * state.level;
    state.lines += cleared;
    updateLevel();
  }
  spawnPiece();
  updateStats();
}

function sweepLines() {
  let cleared = 0;
  for (let y = BOARD_ROWS - 1; y >= 0; y -= 1) {
    if (state.board[y].every((cell) => cell)) {
      const row = state.board.splice(y, 1)[0];
      row.fill(null);
      state.board.unshift(row);
      cleared += 1;
      y += 1;
    }
  }
  return cleared;
}

function updateLevel() {
  const level = Math.floor(state.lines / 10) + 1;
  if (level !== state.level) {
    state.level = level;
    state.dropInterval = Math.max(
      MIN_DROP_INTERVAL,
      BASE_DROP_INTERVAL - (level - 1) * 70
    );
    state.dropCounter = 0;
  }
}

function updateStats() {
  scoreEl.textContent = state.score.toLocaleString();
  linesEl.textContent = state.lines.toString();
  levelEl.textContent = state.level.toString();
}

function updateStatus(mode, message) {
  statusPill.classList.remove(
    "status-pill--idle",
    "status-pill--running",
    "status-pill--paused"
  );
  const textMap = {
    idle: message || "ready",
    running: "playing",
    paused: "paused",
  };
  statusPill.textContent = textMap[mode] || "ready";
  statusPill.classList.add(`status-pill--${mode}`);
}

function draw() {
  boardCtx.clearRect(0, 0, boardCanvas.width, boardCanvas.height);
  drawBackground(boardCtx, boardCanvas.width, boardCanvas.height);
  drawMatrix(boardCtx, state.board, { x: 0, y: 0 });
  if (state.piece) {
    drawMatrix(boardCtx, state.piece.matrix, state.piece.position, state.piece.color);
  }
}

function drawBackground(ctx, width, height) {
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "rgba(255, 255, 255, 0.9)");
  gradient.addColorStop(1, "rgba(255, 247, 226, 0.9)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= BOARD_COLS; x += 1) {
    ctx.beginPath();
    ctx.moveTo(x * cellSize, 0);
    ctx.lineTo(x * cellSize, height);
    ctx.stroke();
  }
  for (let y = 0; y <= BOARD_ROWS; y += 1) {
    ctx.beginPath();
    ctx.moveTo(0, y * cellSize);
    ctx.lineTo(width, y * cellSize);
    ctx.stroke();
  }
}

function drawMatrix(ctx, matrix, offset, forcedColor) {
  matrix.forEach((row, y) => {
    row.forEach((value, x) => {
      if (!value) return;
      const color = forcedColor || value;
      ctx.fillStyle = color;
      const px = (x + offset.x) * cellSize;
      const py = (y + offset.y) * cellSize;
      ctx.fillRect(px, py, cellSize, cellSize);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.65)";
      ctx.lineWidth = 2;
      ctx.strokeRect(px + 1, py + 1, cellSize - 2, cellSize - 2);
    });
  });
}

function drawNext() {
  nextCtx.clearRect(0, 0, nextCanvas.width, nextCanvas.height);
  nextCtx.fillStyle = "#fff9fd";
  nextCtx.fillRect(0, 0, nextCanvas.width, nextCanvas.height);
  if (!state.nextPiece) return;
  const matrix = state.nextPiece.matrix;
  const cell = Math.floor(Math.min(nextCanvas.width, nextCanvas.height) / 5);
  const startX = (nextCanvas.width - matrix[0].length * cell) / 2;
  const startY = (nextCanvas.height - matrix.length * cell) / 2;
  matrix.forEach((row, y) => {
    row.forEach((value, x) => {
      if (!value) return;
      nextCtx.fillStyle = state.nextPiece.color;
      const px = startX + x * cell;
      const py = startY + y * cell;
      nextCtx.fillRect(px, py, cell, cell);
      nextCtx.strokeStyle = "rgba(255, 255, 255, 0.65)";
      nextCtx.lineWidth = 2;
      nextCtx.strokeRect(px + 1, py + 1, cell - 2, cell - 2);
    });
  });
}

function handleKeydown(event) {
  const { key } = event;
  if (key === "Enter" && !state.running) {
    event.preventDefault();
    startGame();
    return;
  }

  if (!state.running) return;

  if (["ArrowLeft", "ArrowRight", "ArrowDown", "ArrowUp", " ", "Space"].includes(key)) {
    event.preventDefault();
  }

  if (state.paused && key !== "p" && key !== "P") {
    return;
  }

  switch (key) {
    case "ArrowLeft":
      movePiece(-1, 0);
      break;
    case "ArrowRight":
      movePiece(1, 0);
      break;
    case "ArrowDown":
      if (movePiece(0, 1)) {
        state.score += 1;
        updateStats();
      }
      state.dropCounter = 0;
      break;
    case "ArrowUp":
      rotatePiece();
      break;
    case " ":
    case "Space":
      hardDrop();
      updateStats();
      break;
    case "p":
    case "P":
      if (state.paused) {
        resumeGame();
      } else {
        pauseGame();
      }
      break;
    default:
      break;
  }
}

function setupUI() {
  toggleBtn.addEventListener("click", () => {
    if (!state.running) {
      startGame();
    } else if (state.paused) {
      resumeGame();
    } else {
      pauseGame();
    }
  });
}

function setButtonPressed(active) {
  toggleBtn.setAttribute("aria-pressed", active ? "true" : "false");
}

document.addEventListener("keydown", handleKeydown);
setupUI();
draw();
updateStatus("idle");
setButtonPressed(false);
