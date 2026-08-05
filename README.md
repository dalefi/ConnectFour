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

## Training starten

Alle Befehle vom Projekt-Root aus, damit `src.*` importierbar ist.

1. Abhängigkeiten. Erst torch vom PyTorch-Index, dann der Rest von PyPI:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

```bash
pip install -r requirements.txt
```

`--index-url` **ersetzt** PyPI, statt es zu ergänzen — auf dem PyTorch-Index
liegen nur torch-Pakete. Deshalb zwei Befehle: der zweite lässt das bereits
installierte CUDA-torch in Ruhe und holt nur die übrigen Pakete.

Ohne CUDA reicht `pip install -r requirements.txt` allein (torch kommt dann als
CPU-Build von PyPI). Der Kanal muss zur Treiberversion passen; `cu121` hat für
Python 3.13 keine Wheels mehr, aktuell sind `cu126` und `cu128`.

2. Postgres hochfahren:

```bash
docker compose up -d
```

3. Tabellen anlegen (löscht vorhandene Tabellen):

```bash
python -m src.database.init_db
```

4. Training starten:

```bash
python -m src.training
```

Ohne Checkpoint fängt der Lauf bei einem frischen Netz an und legt es unter
`accepted_models/cfnet_initial.pt` ab. Um bei einem vorhandenen Modell
weiterzumachen, `generating_model_path` im `__main__`-Block von
`src/training.py` auf den Pfad setzen.

Die alten Checkpoints sind nicht mehr verwendbar: die Netzeingabe ist jetzt
kanonisch (zwei binäre Ebenen aus Sicht des Spielers am Zug, siehe
`encode_board`) statt vorzeichenbehaftetes Brett plus Spielerebene.

### Gegen das Modell spielen

```bash
python -m src.play_against_model
```

### Spielstärke gegen Minimax messen

```bash
python -m src.benchmark_vs_minimax
```

## TODO

- nur stellungen mit gewinnzug ausprobieren
- random position generator statt nur 6 zufällige startzüge spielen
- `src/selfplay.py` ist die alte, sequentielle Fassung und nicht mehr lauffähig
  (`selfplay_parallel.py` hat sie ersetzt) - entweder löschen oder nachziehen
