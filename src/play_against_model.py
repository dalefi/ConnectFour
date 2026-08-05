import asyncio
import os
import queue
import threading

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RadioButtons, Slider, Button

from src.ConnectFour import ConnectFour, Action
from src.CFNet import load_model
from src.NeuralNetBatcher import NeuralNetBatcher
from mcts.searcher.mcts_searcher import mcts_searcher


# ── Modell laden ─────────────────────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
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


# ── KI-Zug: MCTS (async) ─────────────────────────────────────────────────────

async def get_mcts_move(state: ConnectFour) -> int:
    searcher = mcts_searcher(
        iteration_limit=ITERATION_LIMIT,
        batcher=batcher,
        device=DEVICE,
    )
    _, _, mcts_policy = await searcher.search(initial_state=state)
    return int(np.argmax(mcts_policy))


# ── KI-Zug: Minimax (synchron, aber in Thread ausgeführt) ────────────────────

def get_minimax_move(state: ConnectFour, depth: int) -> int:
    return state.get_best_move(depth=depth)


# ── Board zeichnen ────────────────────────────────────────────────────────────

def draw_board(ax, board, status: str = ""):
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

    red_patch    = mpatches.Patch(color="red",    label="KI  (-1 / Rot)")
    yellow_patch = mpatches.Patch(color="yellow", label="Du  ( 1 / Gelb)")
    ax.legend(handles=[yellow_patch, red_patch], loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)

    ax.set_title(status, fontsize=11, pad=6)
    ax.figure.canvas.draw_idle()


# ── Startmenü ─────────────────────────────────────────────────────────────────

class StartMenu:
    """
    Zeigt ein einfaches Matplotlib-Fenster zur Auswahl:
      - Gegner: MCTS-Modell oder Minimax
      - Bei Minimax: Suchtiefe (1–9)
    Gibt die Auswahl per .run() zurück: ("mcts", None) oder ("minimax", depth)
    """

    def __init__(self):
        self.choice  = "mcts"   # default
        self.depth   = 6        # default
        self._done   = False

        self._fig = plt.figure(figsize=(5, 4))
        self._fig.patch.set_facecolor("#1e1e2e")
        self._fig.canvas.manager.set_window_title("Vier Gewinnt – Gegner wählen")

        # Titel
        self._fig.text(0.5, 0.88, "Vier Gewinnt",
                       ha="center", va="center", fontsize=16,
                       color="white", fontweight="bold")
        self._fig.text(0.5, 0.78, "Wähle deinen Gegner:",
                       ha="center", va="center", fontsize=11, color="#cccccc")

        # RadioButtons: Gegnerauswahl
        ax_radio = self._fig.add_axes([0.25, 0.50, 0.50, 0.22],
                                      facecolor="#2a2a3e")
        self._radio = RadioButtons(
            ax_radio,
            labels=("MCTS-Modell", "Minimax"),
            active=0,
            activecolor="#f5a623"
        )
        for lbl in self._radio.labels:
            lbl.set_color("white")
            lbl.set_fontsize(10)
        self._radio.on_clicked(self._on_radio)

        # Slider: Tiefe (nur relevant bei Minimax)
        ax_slider = self._fig.add_axes([0.20, 0.30, 0.60, 0.06],
                                       facecolor="#2a2a3e")
        self._slider = Slider(
            ax_slider, "Tiefe", valmin=1, valmax=9,
            valinit=self.depth, valstep=1,
            color="#f5a623"
        )
        self._slider.label.set_color("white")
        self._slider.valtext.set_color("white")
        self._slider.on_changed(self._on_slider)

        self._depth_hint = self._fig.text(
            0.5, 0.23,
            self._depth_label(self.depth),
            ha="center", va="center", fontsize=8, color="#aaaaaa"
        )

        # Hinweis: Slider nur bei Minimax aktiv
        self._slider_hint = self._fig.text(
            0.5, 0.38, "(Tiefe nur für Minimax relevant)",
            ha="center", va="center", fontsize=8, color="#888888"
        )

        # Start-Button
        ax_btn = self._fig.add_axes([0.30, 0.08, 0.40, 0.10])
        self._btn = Button(ax_btn, "Spiel starten", color="#f5a623", hovercolor="#e09010")
        self._btn.label.set_fontsize(11)
        self._btn.on_clicked(self._on_start)

    def _depth_label(self, d):
        labels = {
            1: "Tiefe 1 – trivial",
            2: "Tiefe 2 – sehr leicht",
            3: "Tiefe 3 – leicht",
            4: "Tiefe 4 – mittel",
            5: "Tiefe 5 – gut (schnell)",
            6: "Tiefe 6 – stark (Standard)",
            7: "Tiefe 7 – sehr stark",
            8: "Tiefe 8 – nahezu perfekt",
            9: "Tiefe 9 – sehr langsam",
        }
        return labels.get(d, "")

    def _on_radio(self, label):
        self.choice = "mcts" if label == "MCTS-Modell" else "minimax"

    def _on_slider(self, val):
        self.depth = int(val)
        self._depth_hint.set_text(self._depth_label(self.depth))
        self._fig.canvas.draw_idle()

    def _on_start(self, event):
        self._done = True
        plt.close(self._fig)

    def run(self):
        plt.show(block=True)
        return self.choice, self.depth


# ── Interaktives Spiel ────────────────────────────────────────────────────────

class InteractiveGame:
    HUMAN = 1
    AI    = -1

    def __init__(self, opponent: str = "mcts", minimax_depth: int = 6):
        """
        opponent      : "mcts" oder "minimax"
        minimax_depth : Suchtiefe (nur relevant bei "minimax")
        """
        self.opponent      = opponent
        self.minimax_depth = minimax_depth

        self.state    = ConnectFour()
        self.done     = False
        self._waiting = False

        self._result_queue = queue.Queue()

        # Asyncio-Loop für MCTS (wird bei Minimax auch gestartet, aber nicht genutzt)
        self._loop   = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        opponent_label = (
            f"Minimax (Tiefe {self.minimax_depth})"
            if opponent == "minimax"
            else "MCTS-Modell"
        )

        self._fig, self._ax = plt.subplots(figsize=(7 * 0.8, 6 * 0.8 + 1.0))
        self._fig.canvas.manager.set_window_title(f"Vier Gewinnt – Gegner: {opponent_label}")
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)

        self._timer = self._fig.canvas.new_timer(interval=50)
        self._timer.add_callback(self._poll_result)
        self._timer.start()

        draw_board(self._ax, self.state.board,
                   f"Dein Zug – Gegner: {opponent_label}")
        plt.show()

        self._timer.stop()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    def _on_click(self, event):
        if self.done or self._waiting or event.inaxes != self._ax or event.xdata is None:
            return

        col   = int(event.xdata)
        valid = [a.target_column for a in self.state.get_possible_actions()]
        if col not in valid:
            draw_board(self._ax, self.state.board, "Spalte voll – andere wählen!")
            return

        # ── Mensch spielt ────────────────────────────────────────────────────
        self.state    = self.state.take_action(Action(target_column=col, player=self.HUMAN))
        self._waiting = True
        draw_board(self._ax, self.state.board, "KI denkt …")

        if self._check_end():
            self._waiting = False
            return

        # ── KI-Zug asynchron starten ─────────────────────────────────────────
        state_snapshot = self.state

        if self.opponent == "mcts":
            async def _run_mcts():
                ai_col = await get_mcts_move(state_snapshot)
                self._result_queue.put(ai_col)

            asyncio.run_coroutine_threadsafe(_run_mcts(), self._loop)

        else:  # minimax – läuft in separatem Thread damit GUI nicht blockiert
            depth = self.minimax_depth

            def _run_minimax():
                ai_col = get_minimax_move(state_snapshot, depth)
                self._result_queue.put(ai_col)

            threading.Thread(target=_run_minimax, daemon=True).start()

    def _poll_result(self):
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
        self.done  = True
        winner     = self.state.get_winner()
        msg = {
            self.HUMAN: "Du hast gewonnen! 🎉",
            self.AI:    "KI hat gewonnen!",
            0:          "Unentschieden!",
        }[winner]
        draw_board(self._ax, self.state.board, msg)
        print(msg)
        return True


# ── Einstiegspunkt ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    menu = StartMenu()
    opponent, depth = menu.run()
    InteractiveGame(opponent=opponent, minimax_depth=depth)