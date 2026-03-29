import asyncio
import os
import queue
import threading

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.ConnectFour import ConnectFour, Action
from src.CFNet import load_model
from src.NeuralNetBatcher import NeuralNetBatcher
from mcts.searcher.mcts_searcher import mcts_searcher


# ── Modell laden ─────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
model_output_dir = os.path.join(project_root, "accepted_models")

MODEL_PATH = os.path.join(
    model_output_dir,
    "cfnet_20260215_224459.pt"
)

ITERATION_LIMIT = 400
DEVICE          = "cuda"

model   = load_model(model_path=MODEL_PATH, model_tag=MODEL_PATH)
batcher = NeuralNetBatcher(model, DEVICE, batch_size=1)


# ── KI-Zug (async) ───────────────────────────────────────────────────────────

async def get_ai_move(state: ConnectFour) -> int:
    searcher = mcts_searcher(
        iteration_limit=ITERATION_LIMIT,
        batcher=batcher,
        device=DEVICE,
    )
    _, _, mcts_policy = await searcher.search(initial_state=state)
    return int(np.argmax(mcts_policy))


# ── Board auf bestehende ax zeichnen ─────────────────────────────────────────

def draw_board(ax, board, status: str = ""):
    """
    Zeichnet das Board auf eine bestehende ax, ohne ein neues Fenster zu öffnen.
    Ersetzt display_board() für den interaktiven Modus.
    """
    ax.clear()
    ax.set_facecolor("blue")
    ax.set_aspect("equal")

    rows, cols = board.shape

    for r in range(rows):
        for c in range(cols):
            value = board[r, c]
            color = "yellow" if value == 1 else ("red" if value == -1 else "white")
            circle = plt.Circle((c + 0.5, r + 0.5), 0.25, color=color, ec="black")
            ax.add_patch(circle)

    for c in range(cols):
        ax.text(c + 0.5, -0.3, str(c), ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

    ax.set_xlim(0, cols)
    ax.set_ylim(-0.5, rows)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    red_patch = mpatches.Patch(color="red", label="KI  (-1 / Rot)")
    yellow_patch = mpatches.Patch(color="yellow", label="Du  ( 1 / Gelb)")
    ax.legend(handles=[yellow_patch, red_patch], loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)

    ax.set_title(status, fontsize=11, pad=6)
    ax.figure.canvas.draw_idle()


# ── Interaktives Spiel ───────────────────────────────────────────────────────

class InteractiveGame:
    HUMAN = 1
    AI = -1

    def __init__(self):
        self.state = ConnectFour()
        self.done = False
        self._waiting = False

        # KI schreibt fertigen Zug in diese Queue, Polling-Timer liest sie aus
        self._result_queue = queue.Queue()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        self._fig, self._ax = plt.subplots(figsize=(7 * 0.8, 6 * 0.8 + 1.0))
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)

        # Alle 50ms prüfen ob ein KI-Ergebnis vorliegt — läuft im GUI-Thread
        self._timer = self._fig.canvas.new_timer(interval=50)
        self._timer.add_callback(self._poll_result)
        self._timer.start()

        draw_board(self._ax, self.state.board, "Dein Zug – klicke auf eine Spalte")
        plt.show()

        self._timer.stop()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    def _on_click(self, event):
        if self.done or self._waiting or event.inaxes != self._ax or event.xdata is None:
            return

        col = int(event.xdata)
        valid = [a.target_column for a in self.state.get_possible_actions()]
        if col not in valid:
            draw_board(self._ax, self.state.board, "Spalte voll – andere wählen!")
            return

        # ── Mensch spielt ────────────────────────────────────────────────────
        self.state = self.state.take_action(Action(target_column=col, player=self.HUMAN))
        self._waiting = True
        draw_board(self._ax, self.state.board, "KI denkt …")

        if self._check_end():
            self._waiting = False
            return

        # ── KI-Zug asynchron starten ─────────────────────────────────────────
        # Wenn fertig, Ergebnis in die Queue legen – _poll_result holt es ab
        state_snapshot = self.state

        async def _run():
            col = await get_ai_move(state_snapshot)
            self._result_queue.put(col)

        asyncio.run_coroutine_threadsafe(_run(), self._loop)

    def _poll_result(self):
        """Läuft alle 50ms im GUI-Thread. Holt KI-Zug aus der Queue, falls vorhanden."""
        try:
            ai_col = self._result_queue.get_nowait()
        except queue.Empty:
            return

        self.state = self.state.take_action(Action(target_column=ai_col, player=self.AI))
        draw_board(self._ax, self.state.board, f"KI spielte Spalte {ai_col} – dein Zug!")
        self._check_end()
        self._waiting = False

    def _check_end(self) -> bool:
        if not self.state.is_terminal():
            return False
        self.done = True
        winner = self.state.get_winner()
        msg = {self.HUMAN: "Du hast gewonnen!", self.AI: "KI hat gewonnen!", 0: "Unentschieden!"}[winner]
        draw_board(self._ax, self.state.board, msg)
        print(msg)
        return True


if __name__ == "__main__":
    InteractiveGame()