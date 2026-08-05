import asyncio
import os
import numpy as np
from tqdm.asyncio import tqdm

from mcts.searcher.mcts_searcher import mcts_searcher
from src.ConnectFour import ConnectFour, Action
from src.CFNet import load_model
from src.NeuralNetBatcher import NeuralNetBatcher
from src.utils import enable_utf8_console, get_filename


# ── Minimax-Zug (synchron, in Thread ausgeführt) ─────────────────────────────

def get_minimax_move(state: ConnectFour, depth: int) -> int:
    return state.get_best_move(depth=depth)


# ── Temperature-Sampling ──────────────────────────────────────────────────────

def sample_with_temperature(policy: np.ndarray, temperature: float) -> int:
    """
    Samplet einen Zug aus der MCTS-Policy mit gegebener Temperatur.
      temperature → 0 : nähert sich argmax (deterministisch)
      temperature = 1 : proportional zur Policy
      temperature > 1 : gleichmäßiger / explorativer
    """
    log_p = np.log(policy + 1e-8) / temperature
    p = np.exp(log_p - np.max(log_p))  # numerisch stabil
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))


# ── Ein Spiel: Modell vs. Minimax ─────────────────────────────────────────────

async def play_one_game(
    game_id: int,
    start_state: ConnectFour,  # vorbereiteter Startzustand (bereits kopiert)
    model_player: int,          # 1 oder -1: welcher Spieler ist das Modell
    batcher: NeuralNetBatcher,
    minimax_depth: int,
    iteration_limit: int,
    device: str,
    pbar,
    active_games_status: dict,
    temperature_moves: int,
    temperature: float,
) -> int:
    """
    Spielt ein Spiel ab `start_state` zwischen Modell und Minimax.

    Returns
    -------
    winner : int  (1, -1, oder 0)
    """
    running_state  = start_state
    move_number    = 0
    net_move_count = 0  # zählt nur die eigenen Netz-Züge für Temperature-Cutoff

    # Ein Searcher pro Partie, damit der Teilbaum zwischen den Zuegen erhalten bleibt.
    searcher = mcts_searcher(
        iteration_limit=iteration_limit,
        batcher=batcher,
        device=device,
    )

    while not running_state.is_terminal():
        current_player = running_state.get_current_player()

        if current_player == model_player:
            # ── Modell-Zug (MCTS, async) ──────────────────────────────────
            active_games_status[game_id] = f"G{game_id:02d}:NET{move_number:02d}"
            pbar.set_description(
                f"Depth {minimax_depth:2d} [{' | '.join(list(active_games_status.values())[-16:])}]"
            )
            pbar.refresh()

            _, _, mcts_policy = await searcher.search(
                initial_state=running_state,
                add_noise=False,
                temperature=1.0,
            )

            if net_move_count < temperature_moves:
                next_move = sample_with_temperature(mcts_policy, temperature)
            else:
                next_move = int(np.argmax(mcts_policy))

            net_move_count += 1

        else:
            # ── Minimax-Zug (sync → Thread, damit Event-Loop nicht blockiert) ──
            active_games_status[game_id] = f"G{game_id:02d}:MM{move_number:02d}"
            pbar.set_description(
                f"Depth {minimax_depth:2d} [{' | '.join(list(active_games_status.values())[-16:])}]"
            )
            pbar.refresh()

            state_snapshot = running_state
            loop = asyncio.get_event_loop()
            next_move = await loop.run_in_executor(
                None, get_minimax_move, state_snapshot, minimax_depth
            )

        running_state.make_move(next_move)
        move_number += 1

    return running_state.get_winner()


# ── Benchmark gegen eine bestimmte Tiefe ─────────────────────────────────────

async def benchmark_one_depth(
    batcher: NeuralNetBatcher,
    minimax_depth: int,
    num_pairs: int,
    iteration_limit: int,
    device: str,
    random_opening_moves: int,
    temperature_moves: int,
    temperature: float,
) -> dict:
    """
    Spielt `num_pairs` Paare gegen Minimax(depth).

    Jedes Paar teilt dieselbe zufällige Startposition:
      - Spiel A: Modell ist Spieler 1 (beginnt)
      - Spiel B: Minimax ist Spieler 1 (beginnt), Modell ist -1

    → Der Startvorteil hebt sich innerhalb jedes Paares perfekt auf.
    Gesamtspiele = num_pairs * 2.

    Returns
    -------
    dict mit wins_model, wins_minimax, draws, total_games, win_rate_model
    """
    active_games_status = {}
    statistics = {"wins_model": 0, "wins_minimax": 0, "draws": 0}
    stats_lock = asyncio.Lock()
    semaphore  = asyncio.Semaphore(16)
    total_games = num_pairs * 2
    pbar = tqdm(total=total_games, desc=f"Depth {minimax_depth:2d}", leave=True)

    # Paare vorbereiten: für jedes Paar eine Startposition generieren,
    # dann zwei Tasks erstellen (Modell beginnt / Minimax beginnt)
    tasks = []
    for pair_idx in range(num_pairs):
        start_state = ConnectFour.random_start_state(
            max_random_moves=random_opening_moves
        )

        for game_in_pair, model_player in enumerate([1, -1]):
            game_id = pair_idx * 2 + game_in_pair

            # Jedes Spiel bekommt seine eigene Kopie des Startzustands
            state_copy = ConnectFour(
                board=start_state.board.copy(),
                currentPlayer=start_state.currentPlayer,
                last_move=start_state.last_move,
            )

            async def sem_task(gid=game_id, state=state_copy, mp=model_player):
                async with semaphore:
                    winner = await play_one_game(
                        game_id=gid,
                        start_state=state,
                        model_player=mp,
                        batcher=batcher,
                        minimax_depth=minimax_depth,
                        iteration_limit=iteration_limit,
                        device=device,
                        pbar=pbar,
                        active_games_status=active_games_status,
                        temperature_moves=temperature_moves,
                        temperature=temperature,
                    )

                async with stats_lock:
                    if winner == 0:
                        statistics["draws"] += 1
                    elif winner == mp:
                        statistics["wins_model"] += 1
                    else:
                        statistics["wins_minimax"] += 1

                    pbar.set_postfix({
                        "Model": statistics["wins_model"],
                        "MM":    statistics["wins_minimax"],
                        "Draw":  statistics["draws"],
                    })
                pbar.update(1)

            tasks.append(sem_task())

    await asyncio.gather(*tasks)
    pbar.close()

    win_rate = statistics["wins_model"] / total_games
    return {**statistics, "total_games": total_games, "win_rate_model": win_rate}


# ── Haupt-Benchmark ───────────────────────────────────────────────────────────

async def run_benchmark(
    model_path: str,
    depths: list[int],
    num_pairs_per_depth: int   = 25,
    iteration_limit: int       = 400,
    device: str                = "cuda",
    random_opening_moves: int  = 4,
    temperature_moves: int     = 3,
    temperature: float         = 1.0,
):
    """
    Benchmarkt ein trainiertes Modell gegen Minimax mit verschiedenen Tiefen.

    Parameters
    ----------
    model_path           : Pfad zum .pt-Modell
    depths               : Liste von Minimax-Tiefen, z.B. [2, 3, 4, 5, 6, 7, 8]
    num_pairs_per_depth  : Anzahl Startposition-Paare pro Tiefe.
                           Gesamtspiele pro Tiefe = num_pairs * 2.
    iteration_limit      : MCTS-Iterationen pro Zug
    device               : "cuda" oder "cpu"
    random_opening_moves : Max. zufällige Züge zur Eröffnung (via random_start_state).
                           0 = immer leeres Brett
    temperature_moves    : Für wie viele eigene Netz-Züge Temperature-Sampling
                           genutzt wird (danach argmax). 0 = immer argmax.
    temperature          : Temperature-Wert (1.0 = proportional zur Policy)
    """
    enable_utf8_console()

    model_name = get_filename(model_path)
    model      = load_model(model_path=model_path, model_tag=model_name)
    batcher    = NeuralNetBatcher(model, device, batch_size=16)

    print("\n" + "=" * 66)
    print(f"  Benchmark:         {model_name}")
    print(f"  Tiefen:            {depths}")
    print(f"  Paare / Tiefe:     {num_pairs_per_depth}  →  {num_pairs_per_depth * 2} Spiele")
    print(f"  MCTS-Its:          {iteration_limit}")
    print(f"  Zuf. Eröffnung:    max. {random_opening_moves} Züge")
    print(f"  Temperature:       {temperature} für erste {temperature_moves} Netz-Züge")
    print("=" * 66 + "\n")

    all_results = {}

    for depth in depths:
        results = await benchmark_one_depth(
            batcher=batcher,
            minimax_depth=depth,
            num_pairs=num_pairs_per_depth,
            iteration_limit=iteration_limit,
            device=device,
            random_opening_moves=random_opening_moves,
            temperature_moves=temperature_moves,
            temperature=temperature,
        )
        all_results[depth] = results

        print(
            f"  Tiefe {depth:2d} │ "
            f"Modell: {results['wins_model']:3d}W  "
            f"Minimax: {results['wins_minimax']:3d}W  "
            f"Remis: {results['draws']:3d}  │  "
            f"Win-Rate Modell: {results['win_rate_model']:.1%}"
        )

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  ZUSAMMENFASSUNG")
    print("=" * 66)
    print(f"  {'Tiefe':>6}  {'Spiele':>6}  {'Modell-W':>8}  {'MM-W':>6}  {'Remis':>6}  {'Win-Rate':>9}")
    print("  " + "-" * 54)
    for depth, r in all_results.items():
        print(
            f"  {depth:6d}  {r['total_games']:6d}  {r['wins_model']:8d}"
            f"  {r['wins_minimax']:6d}  {r['draws']:6d}  {r['win_rate_model']:9.1%}"
        )
    print("=" * 66 + "\n")

    return all_results


# ── Einstiegspunkt ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir         = os.path.dirname(os.path.abspath(__file__))
    project_root       = os.path.dirname(script_dir)
    accepted_model_dir = os.path.join(project_root, "accepted_models")

    # ── Konfiguration ─────────────────────────────────────────────────────────
    MODEL_PATH           = os.path.join(accepted_model_dir, "cfnet_20260215_224459.pt")
    DEPTHS               = [2, 3, 4, 5, 6, 7, 8]
    NUM_PAIRS_PER_DEPTH  = 25    # → 50 Spiele pro Tiefe
    ITERATION_LIMIT      = 400
    DEVICE               = "cuda"
    RANDOM_OPENING_MOVES = 4     # 0 = leeres Brett, 4 = empfohlen
    TEMPERATURE_MOVES    = 3     # erste 3 Netz-Züge per Sampling
    TEMPERATURE          = 1.0
    # ─────────────────────────────────────────────────────────────────────────

    asyncio.run(run_benchmark(
        model_path=MODEL_PATH,
        depths=DEPTHS,
        num_pairs_per_depth=NUM_PAIRS_PER_DEPTH,
        iteration_limit=ITERATION_LIMIT,
        device=DEVICE,
        random_opening_moves=RANDOM_OPENING_MOVES,
        temperature_moves=TEMPERATURE_MOVES,
        temperature=TEMPERATURE,
    ))