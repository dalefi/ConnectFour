"""
Beispielpartien der akzeptierten Modelle gegen sich selbst - fuer den Vortrag und
fuer die Analyse des Trainingsverlaufs.

Alle Modelle spielen dieselben, geseedeten Startstellungen. Nur dadurch ist
"gleiche Stellung, verschiedenes Modell" ueberhaupt vergleichbar - und genau das
macht den Fortschritt sichtbar. Eine Selfplay-Partie allein zeigt ihn nicht: der
Gegner skaliert mit, deshalb sieht die Partie eines schwachen Modells aus wie die
eines starken.

Aufruf vom Projekt-Root:

    python -m src.example_games
"""
import argparse
import asyncio
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mcts.searcher.mcts_searcher import mcts_searcher
from src.CFNet import load_model
from src.ConnectFour import ConnectFour
from src.NeuralNetBatcher import NeuralNetBatcher
from src.utils import temperature_for_move

SCHEMA_VERSION = 1
START_POSITION_SEED = 20260905
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Der Batcher verarbeitet einen Stapel, sobald batch_size Anfragen anliegen -
# sonst erst nach diesem Timeout. Hier laufen genau num_games Partien und keine
# neuen kommen nach: sobald die ersten fertig sind, liegen dauerhaft weniger
# Anfragen an als batch_size, und jede Auswertung zahlt den vollen Timeout.
# Bei 400 Iterationen je Zug waeren das mit dem Standardwert 20 ms rund 8 s pro
# Zug. Deshalb hier deutlich kuerzer; solange genug Partien laufen, bilden sich
# die Stapel trotzdem ueber batch_size.
BATCHER_TIMEOUT = 0.002


# ---------------------------------------------------------------------------
# Startstellungen
# ---------------------------------------------------------------------------

def _position_record(state, num_random_moves, index):
    return {
        "index": index,
        "board": state.board.tolist(),
        "current_player": int(state.get_current_player()),
        "num_random_moves": num_random_moves,
    }


def make_start_positions(count, max_random_moves=4, seed=START_POSITION_SEED):
    """
    `count` reproduzierbare, paarweise verschiedene, nicht beendete Stellungen.

    Index 0 ist immer das leere Brett - die interessanteste einzelne
    Vergleichsstellung. Der Rest entsteht aus 1..max_random_moves uniform
    zufaelligen Zuegen.

    Bewusst nicht ConnectFour.random_start_state: die zieht aus dem globalen
    np.random und liefert damit bei jedem Modell andere Stellungen.

    max_random_moves ist absichtlich kleiner als die 6 der Trainingsdaten-
    erzeugung. Die Zufallszuege sind keine Modellzuege; je mehr davon im Brett
    stehen, desto mehr Unsinn wird dem Modell im Vortrag zugeschrieben.
    """
    rng = np.random.RandomState(seed)

    positions = [_position_record(ConnectFour(), 0, 0)]
    seen = {json.dumps(positions[0]["board"])}
    attempts = 0

    while len(positions) < count:
        attempts += 1
        if attempts > 200 * count:
            raise RuntimeError(
                f"Nur {len(positions)} von {count} verschiedenen Startstellungen "
                f"mit max_random_moves={max_random_moves} gefunden. "
                f"max_random_moves erhoehen oder count senken."
            )

        num_moves = int(rng.randint(1, max_random_moves + 1))
        state = ConnectFour()
        for _ in range(num_moves):
            columns = [action.target_column for action in state.get_possible_actions()]
            state.make_move(int(columns[rng.randint(len(columns))]))
            if state.is_terminal():
                break

        if state.is_terminal():
            continue

        key = json.dumps(state.board.tolist())
        if key in seen:
            continue

        seen.add(key)
        positions.append(_position_record(state, num_moves, len(positions)))

    return positions


# ---------------------------------------------------------------------------
# Reihenfolge der Modelle
# ---------------------------------------------------------------------------

def model_generation_order(paths):
    """
    Modelle in Trainingsreihenfolge: cfnet_initial zuerst, danach nach dem
    Zeitstempel im Dateinamen.

    Alphabetisch sortiert 'cfnet_initial' hinter die Zeitstempel ('i' > '2') -
    der Dateiname darf die Reihenfolge also nicht bestimmen.
    """
    def sort_key(path):
        stem = Path(path).stem
        if stem == "cfnet_initial":
            return (0, "")
        timestamp = re.fullmatch(r"cfnet_(\d{8}_\d{6})", stem)
        if timestamp is not None:
            return (1, timestamp.group(1))
        return (2, stem)

    return sorted((Path(path) for path in paths), key=sort_key)


# ---------------------------------------------------------------------------
# Eine Partie aufzeichnen
# ---------------------------------------------------------------------------

def _rounded(values):
    # 5 Stellen reichen fuer die Anzeige und halbieren die Dateigroesse.
    return [round(float(value), 5) for value in values]


async def play_example_game(batcher, start_position, iteration_limit,
                            use_training_temperature=False):
    """
    Eine vollstaendig aufgezeichnete Partie eines Modells gegen sich selbst.

    Kein Dirichlet-Rauschen: das ist beim Erzeugen von Trainingsdaten zwingend,
    macht das Modell hier aber absichtlich schwaecher. Gezeigt werden soll die
    Spielstaerke, die Vielfalt kommt aus den Startstellungen. Aus demselben Grund
    wird standardmaessig greedy gezogen; use_training_temperature schaltet auf
    die Temperaturverteilung des Trainings um.

    Alle Bewertungen sind aus Sicht des Spielers am Zug - so, wie das Netz sie
    ausgibt (siehe encode_board). Die auf Spieler 1 normierte Kurve leitet der
    Viewer daraus ab, sie wird nicht mitgespeichert.
    """
    state = ConnectFour(
        board=np.array(start_position["board"], dtype=np.int8),
        currentPlayer=start_position["current_player"],
    )
    # Ein Searcher fuer die ganze Partie: der Teilbaum bleibt ueber die Zuege
    # hinweg erhalten, statt jedes Mal neu aufgebaut zu werden.
    searcher = mcts_searcher(iteration_limit=iteration_limit, batcher=batcher)
    moves = []

    while not state.is_terminal():
        move_number = len(moves)

        nn_eval, nn_policy, mcts_policy = await searcher.search(
            initial_state=state,
            add_noise=False,
            temperature=1.0,
        )

        temperature = temperature_for_move(move_number) if use_training_temperature else 0.0
        selection_policy = searcher.get_policy_from_child_visits(temperature=temperature)
        played_move = int(np.random.choice(7, p=selection_policy))

        moves.append({
            "move_number": move_number,
            "board": state.board.tolist(),
            "current_player": int(state.get_current_player()),
            "nn_eval": round(float(nn_eval), 5),
            "mcts_value": round(float(searcher.get_root_value()), 5),
            "nn_policy": _rounded(nn_policy),
            "mcts_policy": _rounded(mcts_policy),
            "visits": [int(count) for count in searcher.get_root_visits()],
            "played_move": played_move,
            "solver_scores": None,
        })

        state.make_move(played_move)

    return {
        "start_position": start_position,
        "moves": moves,
        "outcome": {
            "winner": int(state.get_winner()),
            "num_moves": len(moves),
        },
    }


# ---------------------------------------------------------------------------
# Ablage
# ---------------------------------------------------------------------------

def write_game(model_dir, game_index, game, model, config):
    """Eine Partie als selbsttragende JSON-Datei."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    path = model_dir / f"game_{game_index:02d}.json"
    path.write_text(
        json.dumps({
            "schema_version": SCHEMA_VERSION,
            "model": model,
            "config": config,
            **game,
        }),
        encoding="utf-8",
    )
    return path


def _git_commit():
    """Damit ein Artefakt einem Codestand zuordenbar bleibt."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=PROJECT_ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def write_manifest(output_dir, start_positions, models, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "manifest.json"
    path.write_text(
        json.dumps({
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "config": config,
            "start_positions": start_positions,
            "models": models,
        }, indent=2),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Generierung
# ---------------------------------------------------------------------------

async def generate_for_model(model_path, generation, start_positions, output_dir,
                             iteration_limit, max_parallel, device, config):
    """
    Alle Partien eines Modells, gleichzeitig gespielt.

    Ein Batcher fuer alle Partien, batch_size gleich der Zahl gleichzeitiger
    Partien: nur so laufen die Netzauswertungen wirklich gebuendelt statt
    einzeln. Zum kurzen Timeout siehe BATCHER_TIMEOUT.
    """
    tag = model_path.stem
    model = load_model(model_path=str(model_path), model_tag=tag)
    parallel = min(max_parallel, len(start_positions))
    batcher = NeuralNetBatcher(model, device, batch_size=parallel,
                               timeout=BATCHER_TIMEOUT)
    semaphore = asyncio.Semaphore(parallel)

    model_info = {
        "tag": tag,
        "path": Path(model_path).as_posix(),
        "generation": generation,
    }

    progress = tqdm(total=len(start_positions), desc=f"  {tag}",
                    unit="Partie", leave=False)

    async def one(position):
        async with semaphore:
            game = await play_example_game(
                batcher,
                position,
                iteration_limit,
                use_training_temperature=config["move_selection"] == "training_temperature",
            )
        progress.update(1)
        return game

    games = await asyncio.gather(*(one(position) for position in start_positions))
    progress.close()

    for index, game in enumerate(games):
        write_game(Path(output_dir) / tag, index, game, model_info, config)

    return model_info


async def generate(models_dir, output_dir, num_games, iteration_limit, max_parallel,
                   max_random_moves, seed, use_training_temperature, device):
    model_paths = model_generation_order(Path(models_dir).glob("*.pt"))
    if not model_paths:
        raise SystemExit(f"Keine Modelle in {models_dir} gefunden.")

    start_positions = make_start_positions(num_games, max_random_moves, seed)
    config = {
        "iteration_limit": iteration_limit,
        "add_noise": False,
        "move_selection": "training_temperature" if use_training_temperature else "greedy",
        "seed": seed,
        "max_random_moves": max_random_moves,
    }

    print(f"{len(model_paths)} Modelle x {num_games} Partien auf {device}, "
          f"{iteration_limit} MCTS-Iterationen je Zug")

    models = []
    for generation, model_path in enumerate(tqdm(model_paths, desc="Modelle", unit="Modell")):
        models.append(await generate_for_model(
            model_path, generation, start_positions, output_dir,
            iteration_limit, max_parallel, device, config,
        ))
        # Nach jedem Modell neu schreiben: ein abgebrochener Lauf hinterlaesst
        # dann trotzdem ein benutzbares Manifest.
        write_manifest(output_dir, start_positions, models, config)

    return Path(output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--num-games", type=int, default=10,
                        help="Partien je Modell (= Zahl der Startstellungen)")
    parser.add_argument("--iteration-limit", type=int, default=400,
                        help="MCTS-Iterationen je Zug, wie im Training")
    parser.add_argument("--max-parallel", type=int, default=16,
                        help="gleichzeitige Partien je Modell")
    parser.add_argument("--max-random-moves", type=int, default=4,
                        help="Zufallszuege in den Startstellungen")
    parser.add_argument("--seed", type=int, default=START_POSITION_SEED)
    parser.add_argument("--training-temperature", action="store_true",
                        help="Zugauswahl wie im Training statt greedy")
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "accepted_models"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "example_games"))
    args = parser.parse_args()

    output_dir = asyncio.run(generate(
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        num_games=args.num_games,
        iteration_limit=args.iteration_limit,
        max_parallel=args.max_parallel,
        max_random_moves=args.max_random_moves,
        seed=args.seed,
        use_training_temperature=args.training_temperature,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ))

    print(f"\nFertig: {output_dir}")
    print("Viewer bauen:  python -m src.example_games_viewer")


if __name__ == "__main__":
    main()
