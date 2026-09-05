# Beispielpartien der akzeptierten Modelle + Viewer

Stand: 2026-09-05

## Zweck

Zwei Adressaten, ein Artefakt:

1. **Vortrag** - zeigen, wie die Modelle aus `accepted_models/` im Laufe des
   Trainings besser werden.
2. **Analyse** - nachvollziehen, was Netz und Suche in einer konkreten Stellung
   jeweils beitragen.

## Die tragende Entscheidung

Selfplay-Partien zeigen Fortschritt **strukturell nicht**: spielt ein Modell
gegen sich selbst, skaliert der Gegner mit, und eine Partie von Modell 3 sieht
aus wie eine von Modell 15. Es fehlt die Bezugsgroesse.

Deshalb spielen **alle Modelle dieselben, geseedeten Startstellungen**. Bei Zug 0
ist das Brett damit ueber alle Modelle identisch - man schaltet das Modell durch
und sieht 17 verschiedene Bewertungen und Policies auf demselben Brett. Das ist
der Fortschritt, sichtbar gemacht.

Ab Zug 1 laufen die Partien auseinander. Nur die Startstellung ist garantiert
vergleichbar; der Viewer sagt das explizit, damit niemand zwei verschiedene
Stellungen fuer zwei Bewertungen derselben haelt.

## Komponenten

### 1. `mcts/searcher/mcts_searcher.py` - zwei Accessoren (additiv)

- `get_root_value()` - `root.totalReward / root.numVisits`, die von der Suche
  verfeinerte Bewertung. Gleiche Vorzeichenkonvention wie `nn_eval`: aus Sicht
  des Spielers am Zug. Die Differenz zu `nn_eval` ist genau der Beitrag der
  Suche - die Pointe von AlphaZero, und das Gegenstueck zu `nn_policy` vs.
  `mcts_policy`.
- `get_root_visits()` - rohe Besuchszahlen je Spalte.

Die Signatur von `search()` bleibt unveraendert. Daran haengen Training,
Selfplay, Solver-Benchmark, Play und die Testsuite; ein Umbau waere reines
Risiko ohne Gegenwert. `get_policy_from_child_visits` benutzt jetzt
`get_root_visits`, statt die Besuchszahlen ein zweites Mal einzusammeln.

### 2. `src/example_games.py` - Generierung

**Startstellungen.** Index 0 ist das leere Brett (die interessanteste einzelne
Vergleichsstellung). Die uebrigen entstehen aus 1..`max_random_moves` uniform
zufaelligen Zuegen, gezogen aus einem `np.random.RandomState(seed)` - also
reproduzierbar und identisch fuer jedes Modell. Duplikate werden verworfen.
`max_random_moves` ist 4, nicht 6 wie in der Trainingsdatenerzeugung: die
Zufallszuege sind keine Modellzuege, und je mehr davon im Brett stehen, desto
mehr Unsinn wird dem Modell im Vortrag zugeschrieben.

**Suche.** `add_noise=False` - Dirichlet-Rauschen ist beim Erzeugen von
Trainingsdaten zwingend, macht das Modell hier aber absichtlich schwaecher.
Zugauswahl greedy (Temperatur 0): die Vielfalt kommt aus den Startstellungen,
gezeigt werden soll die Spielstaerke. `--training-temperature` schaltet auf
`temperature_for_move` um, falls die Trainingsverteilung gewuenscht ist.

**Parallelitaet.** Pro Modell ein `NeuralNetBatcher`, alle Partien gleichzeitig
ueber eine Semaphore - dasselbe Muster wie `selfplay_parallel`. `batch_size` ist
die Zahl gleichzeitiger Partien. Sequentiell waere der Lauf Stunden statt
Minuten.

Zusaetzlich wird der Batcher-Timeout von 20 ms auf 2 ms gesenkt
(`BATCHER_TIMEOUT`). Hier laufen genau `num_games` Partien und es kommen keine
neuen nach: sobald die ersten fertig sind, liegen dauerhaft weniger Anfragen an
als `batch_size`, und jede Auswertung zahlt den vollen Timeout - bei 400
Iterationen je Zug rund 8 s pro Zug. Im Probelauf hat die Senkung die Laufzeit
von 1m31s auf 20s gedrueckt.

**Vorbedingung.** `load_model` lud den Checkpoint ohne `map_location`. Die
Checkpoints in `accepted_models/` wurden unter CUDA gespeichert und liessen sich
auf einem CPU-only-torch daher gar nicht laden - das betraf auch
`play_against_model` und `solver_benchmark`. Behoben, weil dieser Pfad sonst
nicht laeuft.

**Reihenfolge der Modelle.** Explizit ueber einen Generationsindex:
`cfnet_initial` ist 0, der Rest folgt dem Zeitstempel im Dateinamen.
Alphabetisch sortiert `initial` hinter die Zeitstempel - der Dateiname darf die
Reihenfolge also nicht bestimmen.

### 3. Datenformat

`example_games/<modell>/game_NN.json`, eine Datei je Partie, selbsttragend:

```
schema_version, model {tag, path, generation},
config {iteration_limit, add_noise, move_selection, seed, max_random_moves},
start_position {index, board, current_player, num_random_moves},
moves [ ... ], outcome {winner, num_moves}
```

Pro Zug:

| Feld | Bedeutung |
|---|---|
| `move_number` | 0-basiert, ab der Startstellung |
| `board` | Brett **vor** dem Zug |
| `current_player` | Spieler am Zug (1 / -1) |
| `nn_eval` | Value-Head des Netzes, aus Sicht des Spielers am Zug |
| `mcts_value` | Von der Suche verfeinerte Bewertung, gleiche Sicht |
| `nn_policy` | Prior des Netzes, maskiert und normiert |
| `mcts_policy` | Besuchsverteilung bei Temperatur 1 |
| `visits` | Rohe Besuchszahlen |
| `played_move` | Tatsaechlich gespielte Spalte |
| `solver_scores` | `null` - Platzhalter fuer exakte Solver-Bewertungen |

**Vorzeichenkonvention.** Alle Bewertungen sind aus Sicht des Spielers am Zug
gespeichert - so, wie das Netz sie ausgibt (kanonische Kodierung, siehe
`encode_board`). Als Kurve waere das unlesbar, weil das Vorzeichen jeden Zug
kippt. Der Viewer rechnet `nn_eval * current_player` und zeigt die feste
Schach-Konvention: +1 gut fuer Spieler 1, -1 gut fuer Spieler -1. Der
abgeleitete Wert wird **nicht** gespeichert - ein Feld weniger, das
inkonsistent werden koennte.

`example_games/manifest.json` haelt Schema-Version, Zeitpunkt, Git-Commit,
Konfiguration, die Startstellungen und die Modelle samt Generationsindex.

`example_games/` geht mit ins Git: die Vortragsartefakte sollen reproduzierbar
und ohne erneuten Lauf verfuegbar sein. Ausgenommen ist `viewer.html` - die
Datei ist vollstaendig aus den Partien generiert und waere nur eine zweite,
mehrere MB grosse Kopie derselben Daten.

### 4. `src/example_games_viewer.py` - baut `example_games/viewer.html`

Eine eigenstaendige HTML-Datei mit eingebetteten Daten. Kein Server, keine
CORS-Probleme, doppelklickbar und teilbar - im Vortrag deutlich robuster als
"vorher noch einen Server starten".

- **Brett** als SVG. Steine der Startstellung sind visuell abgesetzt und
  beschriftet, damit sie niemand dem Modell anlastet.
- **Navigation.** Pfeil links/rechts: Zug. Pfeil hoch/runter: Modell. Home/End:
  Anfang/Ende. Tasten 0-9: Startstellung. Klick in die Bewertungskurve springt
  zum Zug, Klick in eine Tabellenzeile wechselt das Modell.
- **Pro Zug.** `nn_eval` und `mcts_value` als Balken von -1 bis +1, beschriftet
  mit "aus Sicht des Spielers am Zug". NN-Policy und MCTS-Policy als gruppierte
  Balken ueber Spalte 0-6, gespielter Zug markiert, Besuchszahlen darunter.
- **Bewertungskurve** ueber die Partie, auf Spieler 1 normiert, aktueller Zug
  markiert, klickbar.
- **Startstellungs-Tabelle.** Eine Zeile je Modell mit Generation, `nn_eval`,
  `mcts_value` und bevorzugter Spalte - alles auf der aktuell gewaehlten
  Startstellung. Garantiert vergleichbar, weil die Stellung per Konstruktion
  dieselbe ist. Das ist die "so verbessert sich das Modell"-Ansicht.

## Verworfen

- **Vergleichspanel ueber alle Stellungen.** Sollte die aktuelle Stellung in den
  Partien aller anderen Modelle suchen. Viel Maschinerie (Stellungsindex) fuer
  einen Randnutzen: bei Zug 0 leistet das Modell-Dropdown dasselbe, und tiefer
  treffen sich die Partien selten.
- **Solver-Annotation.** Die exakten Zug-Scores aus `solver_benchmark.py` waeren
  die Grundwahrheit, an der Fortschritt messbar statt nur anschaubar wird.
  Kostet aber 20-60 Minuten fuer ~5000 Stellungen. Das Feld `solver_scores`
  bleibt offen; ein spaeterer, wiederaufnehmbarer Durchlauf kann es fuellen.
- **Umbau von `generate_training_data.play_one_game`.** Etwa 40 Zeilen
  Ueberlappung werden bewusst dupliziert. Die Anforderungen weichen ab (keine
  DB, kein Rauschen, geseedet, mehr Felder), und die Datenerzeugung ist der
  Pfad, den man am wenigsten anfassen will.

## Tests

`tests/test_example_games.py`:

- `make_start_positions` ist deterministisch pro Seed, liefert nicht-terminale,
  duplikatfreie Stellungen mit konsistenten Steinzahlen; Index 0 ist leer.
- Der Generationsindex sortiert `cfnet_initial` nach vorne.
- **Replay-Test:** von `start_position` aus reproduzieren die gespeicherten
  `played_move` jede gespeicherte Brettstellung und das Endergebnis. Faengt
  Off-by-one, Vorzeichenfehler und die Verwechslung von "Brett vor/nach dem Zug".
- Jeder Zug traegt beide Bewertungen und beide Policies, und keine Policy
  empfiehlt eine volle Spalte.
- Viewer-Rauchtest: HTML wird gebaut und enthaelt die eingebetteten Daten.

`tests/test_mcts_searcher.py` (die Accessoren gehoeren zum Searcher, und dort
liegt der `ScriptedEvaluator` schon):

- `get_root_value` entspricht `totalReward / numVisits` und ist auf einer
  Stellung mit sofortigem Gewinnzug deutlich positiv - gleiche Sicht wie
  `nn_eval`, nicht die des Gegners.
- `get_root_visits` ist konsistent mit der zurueckgegebenen `mcts_policy` und
  gibt vollen Spalten 0.

## Verifikation

1. `pytest`
2. Probelauf: 2 Modelle, 2 Partien, `iteration_limit=50` - Schema und Replay pruefen
3. Viewer bauen, im Browser durchsteppen
4. Vollstaendiger Lauf ueber alle Modelle
