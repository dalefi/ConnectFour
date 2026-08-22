"""
Absoluter Massstab fuer die Spielstaerke, gemessen gegen einen perfekten Solver.

Minimax mit fester Tiefe sagt nur "besser als dieser Gegner". Der Solver
(connect-four-ai, Rust, strong-solved) kennt fuer jede Stellung den exakten Wert
jedes Zuges. Damit laesst sich direkt messen, wie oft das Modell optimal zieht
und wie teuer seine Fehler sind - ohne eine einzige Partie zu spielen.

Weil das perfekte Loesen teuer ist (~0.1-3 s pro Stellung), wird ein fester
Stellungssatz einmal geloest und als JSON abgelegt. Danach kostet jede Bewertung
nur noch die Inferenz des Modells selbst.
"""
import asyncio
import json
import os
import time

import numpy as np

from src.ConnectFour import ConnectFour
from src.utils import enable_utf8_console

# torch-abhaengige Importe stehen bewusst in evaluate_model: das Erzeugen des
# Stellungssatzes braucht nur numpy und connect-four-ai und laeuft damit auch in
# einer minimalen Python-3.13-Umgebung ohne torch.

SYMBOL = {0: ".", 1: "x", -1: "o"}


def _require_solver():
    """
    connect-four-ai wird nur zum *Erzeugen* des Stellungssatzes gebraucht.
    Das Bewerten eines Modells liest nur die fertige JSON-Datei und kommt ohne
    das Paket aus - deshalb ist der Import bewusst lokal.
    """
    try:
        import connect_four_ai
    except ImportError as error:
        raise ImportError(
            "connect-four-ai ist nicht installiert. Es gibt nur ein einziges Wheel "
            "(CPython 3.13, Windows x86-64) und kein sdist. "
            "Zum reinen Bewerten wird das Paket nicht gebraucht - dafuer genuegt der "
            "mitgelieferte Stellungssatz unter data/solver_benchmark.json."
        ) from error
    return connect_four_ai


def board_to_position(state):
    """
    ConnectFour -> connect_four_ai.Position.

    Beide lesen das Brett zeilenweise von oben links. Achtung: das Brettstring-
    Format des Solvers ist *kanonisch* - 'x' ist immer der Spieler am Zug, 'o' der
    Gegner. Es ist also nicht Spieler 1, sondern muss wie in encode_board mit dem
    aktuellen Spieler multipliziert werden. Ohne das bekommt der Solver bei jeder
    Stellung mit -1 am Zug die farbvertauschte Stellung - und antwortet still falsch.
    """
    Position = _require_solver().Position

    own_stones = int(np.count_nonzero(state.board == 1))
    opponent_stones = int(np.count_nonzero(state.board == -1))
    difference = own_stones - opponent_stones

    if difference not in (0, 1):
        raise ValueError(
            f"Unerreichbare Stellung: {own_stones} Steine fuer Spieler 1, "
            f"{opponent_stones} fuer Spieler -1"
        )

    expected_player = 1 if difference == 0 else -1
    if state.get_current_player() != expected_player:
        raise ValueError(
            f"Steinzahlen erwarten Spieler {expected_player} am Zug, "
            f"bekommen {state.get_current_player()}"
        )

    canonical = state.board * state.get_current_player()
    return Position.from_board_string(
        "".join(SYMBOL[int(value)] for row in canonical for value in row)
    )


def _move_scores(state, solver=None):
    """
    Exakte Bewertung jedes legalen Zuges; None fuer volle Spalten.

    `solver` sollte ueber mehrere Stellungen hinweg wiederverwendet werden: die
    Transpositionstabelle bleibt dann warm, was das Loesen drastisch beschleunigt.
    """
    if solver is None:
        solver = _require_solver().Solver()
    return solver.get_all_move_scores(board_to_position(state))


def optimal_columns(state) -> set[int]:
    """Alle Spalten, die den bestmoeglichen exakten Ausgang erzwingen."""
    scores = _move_scores(state)
    best = max(score for score in scores if score is not None)
    return {column for column, score in enumerate(scores) if score == best}


def score_loss(state, column) -> int:
    """
    Um wie viel schlechter ist `column` als der beste Zug. 0 = optimal.
    Die Skala ist die des Solvers (Betrag ~ wie schnell gewonnen/verloren wird).
    """
    scores = _move_scores(state)
    if scores[column] is None:
        raise ValueError(f"Spalte {column} ist nicht spielbar")
    best = max(score for score in scores if score is not None)
    return best - scores[column]


# ---------------------------------------------------------------------------
# Stellungssatz aufbauen (einmalig, teuer)
# ---------------------------------------------------------------------------

def build_benchmark_set(path, num_positions=200, min_moves=6, max_moves=20, seed=0):
    """
    Sammelt zufaellige, nicht beendete Stellungen und loest sie exakt.
    Ergebnis wird als JSON abgelegt und muss nur einmal berechnet werden.
    """
    enable_utf8_console()
    rng = np.random.RandomState(seed)
    entries = []

    # Ein Solver fuer alle Stellungen: die Transpositionstabelle bleibt warm.
    solver = _require_solver().Solver()
    started = time.perf_counter()

    while len(entries) < num_positions:
        state = ConnectFour()
        target = rng.randint(min_moves, max_moves + 1)
        for _ in range(target):
            if state.is_terminal():
                break
            columns = [a.target_column for a in state.get_possible_actions()]
            state.make_move(int(columns[rng.randint(len(columns))]))

        if state.is_terminal():
            continue

        scores = _move_scores(state, solver)
        entries.append({
            "board": state.board.tolist(),
            "current_player": int(state.get_current_player()),
            "scores": [None if s is None else int(s) for s in scores],
        })

        if len(entries) % 10 == 0:
            elapsed = time.perf_counter() - started
            rate = elapsed / len(entries)
            print(f"  {len(entries)}/{num_positions} geloest "
                  f"({elapsed:.0f}s, {rate:.1f}s/Stellung, "
                  f"noch ~{rate * (num_positions - len(entries)):.0f}s)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle)

    print(f"Stellungssatz geschrieben: {path} ({len(entries)} Stellungen)")
    return path


def load_benchmark_set(path):
    with open(path, encoding="utf-8") as handle:
        entries = json.load(handle)

    for entry in entries:
        entry["state"] = ConnectFour(
            board=np.array(entry["board"], dtype=np.int8),
            currentPlayer=entry["current_player"],
        )
    return entries


# ---------------------------------------------------------------------------
# Modell bewerten
# ---------------------------------------------------------------------------

async def evaluate_model(model_path, benchmark_path, iteration_limit=400, device="cpu"):
    """
    Misst auf dem vorgeloesten Stellungssatz:
      optimal_rate    - Anteil perfekter Zuege
      mean_score_loss - mittlerer Abstand zum besten Zug
      blunder_rate    - Anteil Zuege, die ein Remis/Sieg in eine Niederlage drehen
    """
    from src.CFNet import load_model
    from src.NeuralNetBatcher import NeuralNetBatcher
    from mcts.searcher.mcts_searcher import mcts_searcher

    entries = load_benchmark_set(benchmark_path)
    model = load_model(model_path=model_path, model_tag=os.path.basename(model_path))
    batcher = NeuralNetBatcher(model, device, batch_size=min(64, len(entries)))

    optimal = 0
    losses = []
    blunders = 0

    async def evaluate_one(entry):
        searcher = mcts_searcher(iteration_limit=iteration_limit, batcher=batcher, device=device)
        _, _, policy = await searcher.search(
            entry["state"], add_noise=False, temperature=0.0
        )
        return entry, int(np.argmax(policy))

    for entry, chosen in await asyncio.gather(*(evaluate_one(e) for e in entries)):
        scores = entry["scores"]
        best = max(score for score in scores if score is not None)
        chosen_score = scores[chosen]

        if chosen_score is None:
            raise RuntimeError(f"Modell waehlte die volle Spalte {chosen}")

        losses.append(best - chosen_score)
        if chosen_score == best:
            optimal += 1
        if best >= 0 > chosen_score:
            blunders += 1

    return {
        "positions": len(entries),
        "optimal_rate": optimal / len(entries),
        "mean_score_loss": float(np.mean(losses)),
        "blunder_rate": blunders / len(entries),
    }


if __name__ == "__main__":
    import torch
    from src.utils import latest_model_path

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    benchmark_path = os.path.join(project_root, "data", "solver_benchmark.json")

    if not os.path.exists(benchmark_path):
        print("Kein Stellungssatz gefunden - wird einmalig erzeugt (dauert einige Minuten):")
        build_benchmark_set(benchmark_path, num_positions=200)

    model_path = latest_model_path(os.path.join(project_root, "accepted_models"))
    if model_path is None:
        raise SystemExit("Kein Modell gefunden. Erst 'python -m src.training' laufen lassen.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = asyncio.run(evaluate_model(model_path, benchmark_path, device=device))

    print(f"\nModell: {os.path.basename(model_path)}")
    print(f"  Stellungen        : {result['positions']}")
    print(f"  Optimale Zuege    : {result['optimal_rate']:.1%}")
    print(f"  Mittlerer Verlust : {result['mean_score_loss']:.2f}")
    print(f"  Grobe Fehler      : {result['blunder_rate']:.1%}  (nicht verloren -> verloren)")
