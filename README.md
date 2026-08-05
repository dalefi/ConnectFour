# ConnectFour

AlphaZero-artiges Training für Vier Gewinnt: ein Policy/Value-Netz (`src/CFNet.py`)
steuert eine MCTS-Suche (`mcts/searcher/mcts_searcher.py`), die Selfplay-Partien
erzeugt; die verbesserten Suchzüge werden zu Trainingszielen. Partien und Züge
liegen in Postgres (`docker-compose.yml`).

## Tests

```bash
pytest
```

`pytest -m "not slow"` überspringt die beiden Trainingsschritt-Tests (~25s).

Die Suite deckt vor allem die Stellen ab, an denen ein Fehler still bleibt:
Gewinnerkennung, Vorzeichen von Reward und Value-Targets, die PUCT-Formel und
die Übereinstimmung von Trainings- und Suchkodierung.

## Ablauf

```bash
python -m src.training
```

## TODO

- nur stellungen mit gewinnzug ausprobieren
- random position generator statt nur 6 zufällige startzüge spielen
- `src/selfplay.py` ist die alte, sequentielle Fassung und nicht mehr lauffähig
  (`selfplay_parallel.py` hat sie ersetzt) - entweder löschen oder nachziehen
