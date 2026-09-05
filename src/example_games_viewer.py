"""
Baut aus den Partien in example_games/ eine eigenstaendige viewer.html.

Die Daten werden eingebettet, nicht nachgeladen: eine Datei, doppelklickbar,
ohne Server und ohne CORS-Aerger - im Vortrag deutlich robuster.

Aufruf vom Projekt-Root:

    python -m src.example_games_viewer
"""
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_games(games_dir):
    """
    Alle Partien, nach Modell-Tag gruppiert und nach Partienummer sortiert.
    Der Index in der Liste ist der Index der Startstellung.
    """
    games_dir = Path(games_dir)
    games = {}

    for model_dir in sorted(path for path in games_dir.iterdir() if path.is_dir()):
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(model_dir.glob("game_*.json"))
        ]
        if records:
            games[model_dir.name] = records

    return games


def build_viewer(games_dir, output_path=None):
    games_dir = Path(games_dir)
    manifest = json.loads((games_dir / "manifest.json").read_text(encoding="utf-8"))
    games = load_games(games_dir)

    if not games:
        raise SystemExit(f"Keine Partien in {games_dir} gefunden.")

    payload = json.dumps({"manifest": manifest, "games": games}, separators=(",", ":"))
    # '<' maskieren, damit ein '</script>' in den Daten das Skript nicht beendet.
    payload = payload.replace("<", "\\u003c")

    output_path = Path(output_path) if output_path else games_dir / "viewer.html"
    output_path.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    return output_path


TEMPLATE = r"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vier Gewinnt - Beispielpartien</title>
<style>
  :root {
    --ink: #16202b;
    --muted: #64748b;
    --line: #d7dee8;
    --panel: #ffffff;
    --ground: #f2f5f9;
    --board: #1f5fbf;
    --p1: #f2c53d;
    --p2: #dc4a3d;
    --nn: #7c8ba1;
    --mcts: #1f5fbf;
    --accent: #0f7b5f;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 18px;
    background: var(--ground); color: var(--ink);
    font: 14px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  h1 { font-size: 19px; margin: 0 0 2px; font-weight: 650; }
  .caveat { color: var(--muted); font-size: 12.5px; margin: 0 0 14px; }
  .panel {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 14px;
  }
  header.panel { margin-bottom: 14px; }
  .controls { display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-end; }
  .field { display: flex; flex-direction: column; gap: 3px; }
  .field label { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
  select {
    font: inherit; padding: 5px 8px; border: 1px solid var(--line);
    border-radius: 6px; background: #fff; color: inherit; min-width: 210px;
  }
  .keys { margin-left: auto; font-size: 12px; color: var(--muted); text-align: right; }
  kbd {
    font: 11px/1 ui-monospace, monospace; border: 1px solid var(--line);
    border-bottom-width: 2px; border-radius: 4px; padding: 2px 5px; background: #fafcfe;
  }

  .layout { display: grid; grid-template-columns: minmax(300px, 400px) 1fr; gap: 14px; align-items: start; }
  @media (max-width: 880px) { .layout { grid-template-columns: 1fr; } }

  .boardwrap { display: flex; flex-direction: column; gap: 10px; align-items: center; }
  .turn { display: flex; align-items: center; gap: 8px; font-size: 13px; align-self: stretch; }
  .dot { width: 14px; height: 14px; border-radius: 50%; border: 1px solid #0003; }
  .nav { display: flex; gap: 6px; align-self: stretch; }
  .nav button {
    font: inherit; flex: 1; padding: 6px 0; border: 1px solid var(--line);
    background: #fff; border-radius: 6px; cursor: pointer; color: inherit;
  }
  .nav button:hover:enabled { background: #eef3f9; }
  .nav button:disabled { opacity: .4; cursor: default; }
  .startnote { font-size: 12px; color: var(--muted); align-self: stretch; }

  .section + .section { margin-top: 14px; }
  .section h2 {
    font-size: 12px; text-transform: uppercase; letter-spacing: .05em;
    color: var(--muted); margin: 0 0 10px; font-weight: 600;
  }
  .hint { font-size: 12px; color: var(--muted); margin: 8px 0 0; }

  .evalrow { display: grid; grid-template-columns: 92px 1fr 62px; gap: 10px; align-items: center; }
  .evalrow + .evalrow { margin-top: 8px; }
  .evalname { font-size: 12.5px; }
  .evaltrack { position: relative; height: 20px; background: #eef2f7; border-radius: 4px; overflow: hidden; }
  .evalfill { position: absolute; top: 0; bottom: 0; border-radius: 3px; }
  .evalzero { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: #b6c2d1; }
  .evalnum { font: 13px ui-monospace, monospace; text-align: right; }

  .policy { display: flex; gap: 6px; align-items: flex-end; }
  .pcol { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .pcol.played .pnum { color: var(--accent); font-weight: 700; }
  .pbars { height: 104px; display: flex; align-items: flex-end; gap: 3px; width: 100%; justify-content: center; }
  .pbar { width: 15px; border-radius: 3px 3px 0 0; min-height: 1px; }
  .pbar.nn { background: var(--nn); }
  .pbar.mcts { background: var(--mcts); }
  .pnum { font: 12px ui-monospace, monospace; }
  .pnum small { color: var(--muted); font-size: 10.5px; display: block; font-family: ui-monospace, monospace; }
  .legend { display: flex; gap: 14px; font-size: 12px; color: var(--muted); margin-top: 8px; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; }

  table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
  th, td { text-align: right; padding: 5px 8px; border-bottom: 1px solid #edf1f6; }
  th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
  th:nth-child(2), td:nth-child(2) { text-align: left; }
  td.num { font-family: ui-monospace, monospace; }
  tr.current { background: #eaf3ff; }
  tr.current td { font-weight: 600; }
  tbody tr { cursor: pointer; }
  tbody tr:hover { background: #f5f8fc; }
</style>
</head>
<body>

<h1>Vier Gewinnt - Beispielpartien der akzeptierten Modelle</h1>
<p class="caveat" id="caveat"></p>

<header class="panel">
  <div class="controls">
    <div class="field">
      <label for="model">Modell (Trainingsreihenfolge)</label>
      <select id="model"></select>
    </div>
    <div class="field">
      <label for="position">Startstellung</label>
      <select id="position"></select>
    </div>
    <div class="keys">
      <kbd>&larr;</kbd> <kbd>&rarr;</kbd> Zug &nbsp;
      <kbd>&uarr;</kbd> <kbd>&darr;</kbd> Modell &nbsp;
      <kbd>0</kbd>-<kbd>9</kbd> Startstellung &nbsp;
      <kbd>Home</kbd> <kbd>End</kbd>
    </div>
  </div>
</header>

<div class="layout">
  <div class="panel boardwrap">
    <div class="turn">
      <span id="turntext"></span>
      <span class="dot" id="turndot"></span>
      <span style="margin-left:auto" id="movecount"></span>
    </div>
    <svg id="board" viewBox="0 0 392 358" width="100%" style="max-width:392px"></svg>
    <div class="nav">
      <button id="first">&laquo;</button>
      <button id="prev">&lsaquo; Zug</button>
      <button id="next">Zug &rsaquo;</button>
      <button id="last">&raquo;</button>
    </div>
    <div class="startnote" id="startnote"></div>
  </div>

  <div>
    <div class="panel section">
      <h2>Bewertung dieser Stellung</h2>
      <div id="evals"></div>
      <p class="hint">+1 heisst gewonnen fuer den Spieler <em>am Zug</em>, -1 verloren.
        Die Differenz zwischen Netz und Suche ist der Beitrag von MCTS.</p>
    </div>

    <div class="panel section">
      <h2>Policy: Netz gegen Suche</h2>
      <div class="policy" id="policy"></div>
      <div class="legend">
        <span><span class="swatch" style="background:var(--nn)"></span>Netz (Prior)</span>
        <span><span class="swatch" style="background:var(--mcts)"></span>Nach MCTS</span>
        <span style="color:var(--accent)">gruen = gespielter Zug</span>
      </div>
    </div>

    <div class="panel section">
      <h2>Bewertungsverlauf (aus Sicht Spieler 1)</h2>
      <svg id="curve" viewBox="0 0 640 190" width="100%"></svg>
    </div>
  </div>
</div>

<div class="panel section" style="margin-top:14px">
  <h2 id="tabletitle"></h2>
  <table>
    <thead><tr>
      <th>Gen</th><th>Modell</th><th>Netz</th><th>MCTS</th>
      <th>Spalte Netz</th><th>Spalte MCTS</th><th>Ausgang</th>
    </tr></thead>
    <tbody id="tablebody"></tbody>
  </table>
  <p class="hint">Alle Modelle auf derselben Startstellung - der einzige garantiert
    vergleichbare Punkt. Zeile anklicken wechselt das Modell.</p>
</div>

<script>
const EXAMPLE_GAMES = __DATA__;
const MODELS = EXAMPLE_GAMES.manifest.models.slice().sort((a, b) => a.generation - b.generation);
const POSITIONS = EXAMPLE_GAMES.manifest.start_positions;
const COLORS = { 1: "var(--p1)", "-1": "var(--p2)" };
const NAMES = { 1: "Spieler 1 (gelb)", "-1": "Spieler -1 (rot)" };

let modelIndex = 0;
let positionIndex = 0;
let moveIndex = 0;

function currentGame() {
  return EXAMPLE_GAMES.games[MODELS[modelIndex].tag][positionIndex];
}

function clampMove() {
  const total = currentGame().moves.length;
  moveIndex = Math.max(0, Math.min(moveIndex, total - 1));
}

// --- Brett ----------------------------------------------------------------

function drawBoard(move, startBoard) {
  const cell = 56, radius = 22;
  const parts = ['<rect x="0" y="0" width="392" height="336" rx="10" fill="var(--board)"/>'];

  for (let row = 0; row < 6; row++) {
    for (let col = 0; col < 7; col++) {
      const value = move.board[row][col];
      const cx = col * cell + cell / 2;
      const cy = row * cell + cell / 2;
      const fill = value === 0 ? "#f7fafc" : COLORS[value];
      // Steine der Startstellung sind Zufallszuege, keine Modellzuege - sie
      // werden abgesetzt, damit niemand sie dem Modell zuschreibt.
      const fromStart = value !== 0 && startBoard[row][col] !== 0;
      const stroke = fromStart ? 'stroke="#12325e" stroke-width="3" stroke-dasharray="4 3"'
                               : 'stroke="#0000001f" stroke-width="1"';
      parts.push(`<circle cx="${cx}" cy="${cy}" r="${radius}" fill="${fill}" ${stroke}/>`);
    }
  }

  // Markierung der Spalte, die als naechstes gespielt wird.
  const cx = move.played_move * cell + cell / 2;
  parts.push(`<polygon points="${cx - 7},4 ${cx + 7},4 ${cx},14" fill="#ffffffcc"/>`);

  // Spaltennummern: sonst muss man zum Zuordnen der Policy-Balken zaehlen.
  for (let col = 0; col < 7; col++) {
    const played = col === move.played_move;
    parts.push(`<text x="${col * cell + cell / 2}" y="353" text-anchor="middle"
      font-size="13" font-weight="${played ? 700 : 400}"
      fill="${played ? "var(--accent)" : "var(--muted)"}">${col}</text>`);
  }

  document.getElementById("board").innerHTML = parts.join("");
}

// --- Bewertungsbalken -----------------------------------------------------

function evalBar(name, value) {
  const magnitude = Math.min(Math.abs(value), 1) * 50;
  const style = value >= 0 ? `left:50%;width:${magnitude}%` : `right:50%;width:${magnitude}%`;
  const color = value >= 0 ? "var(--accent)" : "var(--p2)";
  const sign = value >= 0 ? "+" : "";
  return `<div class="evalrow">
    <span class="evalname">${name}</span>
    <div class="evaltrack">
      <div class="evalfill" style="${style};background:${color}"></div>
      <div class="evalzero"></div>
    </div>
    <span class="evalnum">${sign}${value.toFixed(3)}</span>
  </div>`;
}

// --- Policy ---------------------------------------------------------------

function drawPolicy(move) {
  const peak = Math.max(...move.nn_policy, ...move.mcts_policy, 0.01);
  const columns = [];

  for (let col = 0; col < 7; col++) {
    const nn = move.nn_policy[col], mcts = move.mcts_policy[col];
    const played = col === move.played_move ? " played" : "";
    columns.push(`<div class="pcol${played}">
      <div class="pbars">
        <div class="pbar nn" style="height:${(nn / peak * 100).toFixed(1)}%"></div>
        <div class="pbar mcts" style="height:${(mcts / peak * 100).toFixed(1)}%"></div>
      </div>
      <div class="pnum">${col}
        <small>${(nn * 100).toFixed(0)}% / ${(mcts * 100).toFixed(0)}%</small>
        <small>${move.visits[col]}&#215;</small>
      </div>
    </div>`);
  }

  document.getElementById("policy").innerHTML = columns.join("");
}

// --- Bewertungsverlauf ----------------------------------------------------

function drawCurve(game) {
  const width = 640, height = 190, pad = 26;
  const moves = game.moves;
  const span = Math.max(moves.length - 1, 1);
  const x = i => pad + i / span * (width - 2 * pad);
  // Normierung auf Spieler 1: der Rohwert ist aus Sicht des Spielers am Zug und
  // kippt daher jeden Zug das Vorzeichen - als Kurve unlesbar.
  const y = v => height / 2 - v * (height / 2 - pad);

  const line = key => moves
    .map((move, i) => `${x(i).toFixed(1)},${y(move[key] * move.current_player).toFixed(1)}`)
    .join(" ");

  const parts = [
    `<line x1="${pad}" y1="${height / 2}" x2="${width - pad}" y2="${height / 2}" stroke="#c3cedb"/>`,
    `<text x="4" y="${pad + 4}" font-size="10" fill="var(--muted)">+1</text>`,
    `<text x="4" y="${height - pad + 8}" font-size="10" fill="var(--muted)">-1</text>`,
    `<polyline points="${line("nn_eval")}" fill="none" stroke="var(--nn)" stroke-width="2"/>`,
    `<polyline points="${line("mcts_value")}" fill="none" stroke="var(--mcts)" stroke-width="2.5"/>`,
    `<line x1="${x(moveIndex)}" y1="${pad - 12}" x2="${x(moveIndex)}" y2="${height - pad + 12}"
       stroke="var(--accent)" stroke-width="1.5" stroke-dasharray="3 3"/>`,
  ];

  const outcome = { 1: "Spieler 1 gewinnt", "-1": "Spieler -1 gewinnt", 0: "unentschieden" };
  parts.push(`<text x="${width - pad}" y="14" font-size="11" text-anchor="end"
    fill="var(--muted)">${outcome[game.outcome.winner]} nach ${moves.length} Zuegen</text>`);

  const svg = document.getElementById("curve");
  svg.innerHTML = parts.join("");
  svg.onclick = event => {
    const box = svg.getBoundingClientRect();
    const relative = (event.clientX - box.left) / box.width * width;
    moveIndex = Math.round((relative - pad) / (width - 2 * pad) * span);
    clampMove();
    render();
  };
}

// --- Startstellungs-Tabelle ----------------------------------------------

function drawTable() {
  const rows = MODELS.map((model, index) => {
    const game = EXAMPLE_GAMES.games[model.tag][positionIndex];
    const first = game.moves[0];
    const best = policy => policy.indexOf(Math.max(...policy));
    const outcome = { 1: "1 gewinnt", "-1": "-1 gewinnt", 0: "remis" };

    return `<tr class="${index === modelIndex ? "current" : ""}" data-model="${index}">
      <td class="num">${model.generation}</td>
      <td>${model.tag}</td>
      <td class="num">${first.nn_eval.toFixed(3)}</td>
      <td class="num">${first.mcts_value.toFixed(3)}</td>
      <td class="num">${best(first.nn_policy)}</td>
      <td class="num">${best(first.mcts_policy)}</td>
      <td>${outcome[game.outcome.winner]}</td>
    </tr>`;
  });

  const body = document.getElementById("tablebody");
  body.innerHTML = rows.join("");
  body.querySelectorAll("tr").forEach(row => {
    row.onclick = () => { modelIndex = Number(row.dataset.model); clampMove(); render(); };
  });

  const position = POSITIONS[positionIndex];
  document.getElementById("tabletitle").textContent =
    `Startstellung ${positionIndex} - alle Modelle, ${NAMES[position.current_player]} am Zug`;
}

// --- Zusammensetzen -------------------------------------------------------

function render() {
  const game = currentGame();
  const move = game.moves[moveIndex];
  const start = game.start_position;

  document.getElementById("model").value = String(modelIndex);
  document.getElementById("position").value = String(positionIndex);

  document.getElementById("turntext").textContent = `Am Zug: ${NAMES[move.current_player]}`;
  document.getElementById("turndot").style.background = COLORS[move.current_player];
  document.getElementById("movecount").textContent =
    `Zug ${moveIndex + 1} von ${game.moves.length}`;

  document.getElementById("startnote").textContent = start.num_random_moves === 0
    ? "Startstellung: leeres Brett."
    : `Startstellung: ${start.num_random_moves} Zufallszuege (gestrichelt umrandet) - `
      + "keine Modellzuege.";

  drawBoard(move, start.board);
  document.getElementById("evals").innerHTML =
    evalBar("Netz", move.nn_eval) + evalBar("Netz + MCTS", move.mcts_value);
  drawPolicy(move);
  drawCurve(game);
  drawTable();

  document.getElementById("first").disabled = moveIndex === 0;
  document.getElementById("prev").disabled = moveIndex === 0;
  document.getElementById("next").disabled = moveIndex === game.moves.length - 1;
  document.getElementById("last").disabled = moveIndex === game.moves.length - 1;
}

function setup() {
  const manifest = EXAMPLE_GAMES.manifest;
  document.getElementById("caveat").textContent =
    `${MODELS.length} Modelle, ${POSITIONS.length} Startstellungen, `
    + `${manifest.config.iteration_limit} MCTS-Iterationen je Zug. `
    + "Ab Zug 2 laufen die Partien auseinander - nur die Startstellung ist "
    + "ueber alle Modelle identisch.";

  document.getElementById("model").innerHTML = MODELS
    .map((model, index) => `<option value="${index}">Gen ${model.generation} - ${model.tag}</option>`)
    .join("");

  document.getElementById("position").innerHTML = POSITIONS
    .map((position, index) => `<option value="${index}">`
      + `${index}: ${position.num_random_moves} Zufallszuege</option>`)
    .join("");

  document.getElementById("model").onchange = event => {
    modelIndex = Number(event.target.value); clampMove(); render();
  };
  document.getElementById("position").onchange = event => {
    positionIndex = Number(event.target.value); moveIndex = 0; render();
  };

  const step = delta => { moveIndex += delta; clampMove(); render(); };
  const jump = to => { moveIndex = to; clampMove(); render(); };
  document.getElementById("prev").onclick = () => step(-1);
  document.getElementById("next").onclick = () => step(1);
  document.getElementById("first").onclick = () => jump(0);
  document.getElementById("last").onclick = () => jump(1e9);

  const cycleModel = delta => {
    modelIndex = (modelIndex + delta + MODELS.length) % MODELS.length;
    clampMove(); render();
  };

  document.addEventListener("keydown", event => {
    const keys = {
      ArrowLeft: () => step(-1), ArrowRight: () => step(1),
      ArrowUp: () => cycleModel(-1), ArrowDown: () => cycleModel(1),
      Home: () => jump(0), End: () => jump(1e9),
    };
    if (keys[event.key]) { event.preventDefault(); keys[event.key](); return; }
    if (/^[0-9]$/.test(event.key) && Number(event.key) < POSITIONS.length) {
      positionIndex = Number(event.key); moveIndex = 0; render();
    }
  });

  render();
}

setup();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--games-dir", default=str(PROJECT_ROOT / "example_games"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    path = build_viewer(args.games_dir, args.output)
    print(f"Viewer gebaut: {path}")
    print(f"Groesse: {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
