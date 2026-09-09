"""
Das Spielbrett und der Stellungssatz muessen ohne torch importierbar bleiben.

Das ist kein Selbstzweck: die Selfplay-Worker starten als eigene Prozesse, und
das Erzeugen des Solver-Stellungssatzes soll in einer minimalen Umgebung ohne
torch laufen koennen. Frueher zog mcts/base/base.py ueber einen ungenutzten
Import src.utils - und damit torch - in jeden Import des Spielbretts.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_isolated(source):
    """Fuehrt Code in einem frischen Interpreter aus und gibt stdout zurueck."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_importing_the_board_does_not_pull_in_torch():
    output = run_isolated("""
        import sys
        sys.path.insert(0, '.')
        from src.ConnectFour import ConnectFour
        print('torch' in sys.modules)
    """)
    assert output == "False", "src.ConnectFour zieht torch mit herein"


def test_importing_utils_does_not_pull_in_torch():
    output = run_isolated("""
        import sys
        sys.path.insert(0, '.')
        import src.utils
        print('torch' in sys.modules)
    """)
    assert output == "False", "src.utils zieht torch mit herein"


def test_the_vendored_mcts_base_does_not_depend_on_project_code():
    output = run_isolated("""
        import sys
        sys.path.insert(0, '.')
        import mcts.base.base
        print('torch' in sys.modules, 'src.utils' in sys.modules)
    """)
    assert output == "False False", "mcts.base.base haengt an Projektcode"


def test_the_board_still_works_without_torch():
    output = run_isolated("""
        import sys
        sys.path.insert(0, '.')
        from src.ConnectFour import ConnectFour
        state = ConnectFour()
        for column in (0, 1, 0, 1, 0, 1, 0):
            state.make_move(column)
        print(state.get_winner(), state.is_terminal(), 'torch' in sys.modules)
    """)
    assert output == "1 True False"
