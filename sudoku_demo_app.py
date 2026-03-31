from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

from kona.sudoku import (
    SolveResult,
    benchmark_hard_puzzles,
    board_to_text,
    parse_puzzle,
    random_hard_puzzle,
    solve_sudoku,
)


ROOT = Path(__file__).resolve().parent
app = Flask(__name__)


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>kona</title>
  <style>
    body {
      margin: 0;
      font-family: system-ui, sans-serif;
      background: #f5f5f5;
      color: #111;
    }
    .page {
      max-width: 1080px;
      margin: 0 auto;
      padding: 24px;
    }
    .layout {
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 24px;
    }
    .panel {
      background: #fff;
      border: 1px solid #d8d8d8;
      padding: 16px;
    }
    textarea {
      width: 100%;
      min-height: 220px;
      font: 14px/1.4 ui-monospace, monospace;
      box-sizing: border-box;
      margin-bottom: 12px;
    }
    .actions {
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }
    button {
      padding: 8px 12px;
      font: inherit;
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .meta-item {
      background: #fff;
      border: 1px solid #d8d8d8;
      padding: 12px;
    }
    .meta-label {
      font-size: 12px;
      color: #666;
      margin-bottom: 6px;
    }
    .meta-value {
      font-size: 18px;
      font-weight: 600;
    }
    .boards {
      display: grid;
      grid-template-columns: repeat(2, minmax(280px, 1fr));
      gap: 16px;
    }
    .board-wrap {
      background: #fff;
      border: 1px solid #d8d8d8;
      padding: 12px;
    }
    .board-title {
      font-size: 13px;
      color: #666;
      margin-bottom: 8px;
    }
    .board {
      display: grid;
      grid-template-columns: repeat(9, 1fr);
      border: 2px solid #111;
      background: #fff;
      aspect-ratio: 1;
    }
    .cell {
      display: flex;
      align-items: center;
      justify-content: center;
      border-right: 1px solid #cfcfcf;
      border-bottom: 1px solid #cfcfcf;
      font-size: 20px;
      background: #fff;
    }
    .cell.clue {
      font-weight: 700;
    }
    .cell.box-r {
      border-right: 2px solid #111;
    }
    .cell.box-b {
      border-bottom: 2px solid #111;
    }
    .trace {
      margin-top: 16px;
      background: #fff;
      border: 1px solid #d8d8d8;
      padding: 12px;
      white-space: pre-wrap;
      font: 13px/1.4 ui-monospace, monospace;
    }
    .small {
      font-size: 13px;
      color: #555;
      white-space: pre-wrap;
      font-family: ui-monospace, monospace;
    }
    @media (max-width: 900px) {
      .layout {
        grid-template-columns: 1fr;
      }
      .boards {
        grid-template-columns: 1fr;
      }
      .meta {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="layout">
      <div class="panel">
        <textarea id="puzzle" rows="11" cols="24">{{ puzzle }}</textarea>
        <div class="actions">
          <button id="randomBtn">Random</button>
          <button id="benchBtn">Benchmark</button>
          <button id="solveBtn">Solve</button>
        </div>
        <div id="benchmarkSummary" class="small"></div>
      </div>
      <div>
        <div class="meta">
          <div class="meta-item">
            <div class="meta-label">Status</div>
            <div class="meta-value" id="statusValue">Ready</div>
          </div>
          <div class="meta-item">
            <div class="meta-label">Solver</div>
            <div class="meta-value" id="solverValue">kona</div>
          </div>
          <div class="meta-item">
            <div class="meta-label">Time</div>
            <div class="meta-value" id="elapsedValue">-</div>
          </div>
          <div class="meta-item">
            <div class="meta-label">Nodes</div>
            <div class="meta-value" id="nodesValue">-</div>
          </div>
        </div>
        <div class="boards">
          <div class="board-wrap">
            <div class="board-title">Puzzle</div>
            <div id="inputBoard" class="board"></div>
          </div>
          <div class="board-wrap">
            <div class="board-title">Solution</div>
            <div id="outputBoard" class="board"></div>
          </div>
        </div>
        <div id="traceText" class="trace"></div>
      </div>
    </div>
  </div>
  <script>
    const initialBoard = {{ initial_board | tojson }};
    const randomBtn = document.getElementById('randomBtn');
    const benchBtn = document.getElementById('benchBtn');
    const solveBtn = document.getElementById('solveBtn');
    const puzzleEl = document.getElementById('puzzle');

    function renderBoard(target, board, clueBoard) {
      target.innerHTML = '';
      for (let r = 0; r < 9; r++) {
        for (let c = 0; c < 9; c++) {
          const cell = document.createElement('div');
          const value = board?.[r]?.[c] || 0;
          const clue = clueBoard?.[r]?.[c] || 0;
          cell.className = 'cell';
          if (clue) cell.classList.add('clue');
          if ((c + 1) % 3 === 0 && c !== 8) cell.classList.add('box-r');
          if ((r + 1) % 3 === 0 && r !== 8) cell.classList.add('box-b');
          cell.textContent = value ? value : '';
          target.appendChild(cell);
        }
      }
    }

    function renderTrace(trace) {
      document.getElementById('traceText').textContent = trace && trace.length
        ? `energy: ${trace.join(' -> ')}`
        : '';
    }

    function setStatus(result) {
      document.getElementById('statusValue').textContent = result.solved ? 'Solved' : 'Unsolved';
      document.getElementById('solverValue').textContent = result.solved_by;
      document.getElementById('elapsedValue').textContent = `${Math.round(result.elapsed_ms)} ms`;
      document.getElementById('nodesValue').textContent = String(result.guided_nodes ?? '-');
    }

    async function loadRandom() {
      const resp = await fetch('/api/random');
      const data = await resp.json();
      puzzleEl.value = data.puzzle;
      renderBoard(document.getElementById('inputBoard'), data.board, data.board);
      renderBoard(document.getElementById('outputBoard'), data.board, data.board);
      renderTrace([]);
      document.getElementById('statusValue').textContent = 'Ready';
      document.getElementById('solverValue').textContent = 'kona';
      document.getElementById('elapsedValue').textContent = '-';
      document.getElementById('nodesValue').textContent = '-';
      document.getElementById('benchmarkSummary').textContent = '';
    }

    async function runBenchmark() {
      benchBtn.disabled = true;
      randomBtn.disabled = true;
      solveBtn.disabled = true;
      const resp = await fetch('/api/benchmark');
      const data = await resp.json();
      document.getElementById('benchmarkSummary').textContent =
        `solved ${data.solved}/${data.puzzles}\navg ${Math.round(data.avg_elapsed_ms)} ms\nmax ${Math.round(data.max_elapsed_ms)} ms`;
      benchBtn.disabled = false;
      randomBtn.disabled = false;
      solveBtn.disabled = false;
    }

    async function solve() {
      solveBtn.disabled = true;
      randomBtn.disabled = true;
      benchBtn.disabled = true;
      const resp = await fetch('/api/solve', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ puzzle: puzzleEl.value })
      });
      const data = await resp.json();
      if (!resp.ok) {
        document.getElementById('statusValue').textContent = data.error || 'error';
      } else {
        renderBoard(document.getElementById('inputBoard'), data.initial_board, data.initial_board);
        renderBoard(document.getElementById('outputBoard'), data.board, data.initial_board);
        renderTrace(data.energy_trace);
        setStatus(data);
      }
      solveBtn.disabled = false;
      randomBtn.disabled = false;
      benchBtn.disabled = false;
    }

    randomBtn.addEventListener('click', loadRandom);
    benchBtn.addEventListener('click', runBenchmark);
    solveBtn.addEventListener('click', solve);
    renderBoard(document.getElementById('inputBoard'), initialBoard, initialBoard);
    renderBoard(document.getElementById('outputBoard'), initialBoard, initialBoard);
    renderTrace([]);
  </script>
</body>
</html>
"""


def result_to_json(result: SolveResult) -> dict:
    return {
        "solved": result.solved,
        "solved_by": result.solved_by,
        "board": result.board,
        "initial_board": result.initial_board,
        "initial_energy": result.initial_energy,
        "final_energy": result.final_energy,
        "restarts": result.restarts,
        "steps": result.steps,
        "guided_nodes": result.guided_nodes,
        "elapsed_ms": result.elapsed_ms,
        "energy_trace": result.energy_trace,
        "heatmap": result.heatmap,
        "board_text": board_to_text(result.board),
    }


@app.get("/")
def index():
    puzzle = random_hard_puzzle()
    return render_template_string(PAGE, puzzle=puzzle, initial_board=parse_puzzle(puzzle))


@app.get("/favicon.ico")
def favicon():
    return ("", 204)


@app.get("/api/random")
def api_random():
    puzzle = random_hard_puzzle()
    board = parse_puzzle(puzzle)
    return jsonify({"puzzle": puzzle, "board": board})


@app.get("/api/benchmark")
def api_benchmark():
    return jsonify(benchmark_hard_puzzles(seed=0))


@app.post("/api/solve")
def api_solve():
    payload = request.get_json(silent=True) or {}
    puzzle_text = str(payload.get("puzzle") or "").strip()
    if not puzzle_text:
        return jsonify({"error": "Missing puzzle text."}), 400
    try:
        result = solve_sudoku(puzzle_text, seed=0, exact_fallback=True)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result_to_json(result))


if __name__ == "__main__":
    host = os.environ.get("SUDOKU_DEMO_HOST", "0.0.0.0")
    port = int(os.environ.get("SUDOKU_DEMO_PORT", "3010"))
    app.run(host=host, port=port, debug=False)
