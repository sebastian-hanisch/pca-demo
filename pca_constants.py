"""Defaults, Slider-Grenzen und Presets für die PCA-Demo (Hauptkomponentenanalyse an Lieferrouten-Kennzahlen)."""

# --- Merkmale: 12 Kennzahlen je Tour in 4 Gruppen zu je 3 (Name, Einheit, Mittelwert, typische Streuung in Einheiten) ------------
FEATURES = (
    ("Distanz", "m", 45000.0, 15000.0),
    ("Stopps", "Anzahl", 60.0, 20.0),
    ("Ladegewicht", "kg", 1200.0, 400.0),
    ("Zeitfenster-Enge", "min", 90.0, 30.0),
    ("Verspätung", "min", 12.0, 8.0),
    ("Überstunden", "min", 25.0, 15.0),
    ("Fahrzeit je km", "s", 90.0, 25.0),
    ("Stop-and-go-Anteil", "%", 22.0, 10.0),
    ("Parkzeit", "min", 35.0, 12.0),
    ("Retourenquote", "Anteil", 0.06, 0.02),
    ("Sonderwünsche", "Anzahl", 4.0, 2.0),
    ("Zustellversuche", "Anzahl", 1.3, 0.5),
)
N_FEATURES = len(FEATURES)
FEATURE_NAMES = tuple(f[0] for f in FEATURES)
FEATURE_LABELS = tuple(f"{f[0]} [{f[1]}]" for f in FEATURES)
GROUPS = ("Größe", "Zeitdruck", "Verkehr", "Sonderfälle")     # je 3 aufeinanderfolgende Merkmale
GROUP_OF_FEATURE = tuple(i // 3 for i in range(N_FEATURES))

# --- Regler ------------------------------------------------------------------------------------------------------------
DEFAULT_N_TOURS = 300
N_TOURS_MIN, N_TOURS_MAX = 100, 600
DEFAULT_Q = 2
Q_MIN, Q_MAX = 1, 4
DEFAULT_CURVATURE = 0.0
CURVATURE_MIN, CURVATURE_MAX = 0.0, 1.0
DEFAULT_NOISE = 0.25
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_OUTLIER_PCT = 0
OUTLIER_PCT_MIN, OUTLIER_PCT_MAX = 0, 10
DEFAULT_SEED = 7
SCALINGS = ("standardized", "raw")
SCALING_LABELS = {"standardized": "Standardisiert (z-Werte)", "raw": "Rohdaten (Einheiten wie gemessen)"}
DEFAULT_SCALING = "standardized"

# --- Erzeugung ---------------------------------------------------------------------------------------------------------
OUTLIER_SCALE = 10.0                   # Sonderfahrten: latenter Faktor um diesen Faktor vergrößert
CROSS_LOADING = 0.15                   # kleine Querladungen zwischen Merkmalsgruppen
WITHIN_LOADINGS = (0.95, 0.9, 0.85)    # Ladung der drei Merkmale einer Gruppe auf ihren Faktor
CURVATURE_FREQUENCY = 1.6              # Frequenz der sin/cos-Terme der Krümmung
CURVATURE_AMPLITUDE = 2.0              # Länge jeder Spalte der Krümmungsmatrix (in z-Einheiten bei Krümmung 1)
LAYOUT_SEED = 20240915                 # feste Ladungs- und Krümmungsmatrizen (unabhängig vom Seed der Touren)

# --- Auswertung --------------------------------------------------------------------------------------------------------
VARIANCE_TARGET = 0.90                 # "genug" erklärte Varianz für k_90
TRUST_NEIGHBORS = 10                   # k der Trustworthiness
SWEEP_SEEDS = tuple(100_000 + i for i in range(5))                 # feste Sweep-Seeds, unabhängig vom Demo-Seed
SWEEP_CURVATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SWEEP_N_TOURS = 250

# Verdict-Schwellen
UNITS_DOMINANCE = 0.90                 # Anteil von PC1-Varianz, den ein einzelnes Merkmal in Rohdaten trägt
CURVATURE_EXTRA_COMPONENTS = 2         # k_90 mindestens so viel höher als beim geraden Gegenstück
OUTLIER_ANGLE_DEG = 8.0                # PC1 dreht sich durch Ausreißer um mindestens so viel
NOISE_EXTRA_COMPONENTS = 2             # k_90 (gerade) mindestens so viel höher als q

_BASE = {"n_tours": DEFAULT_N_TOURS, "q": 2, "curvature": 0.0, "noise": DEFAULT_NOISE, "outlier_pct": 0, "scaling": "standardized", "seed": DEFAULT_SEED}
PRESETS = {
    "Einfaches Beispiel (scharfer Knick)": {**_BASE},
    "Einheiten-Falle (Rohdaten)": {**_BASE, "scaling": "raw"},
    "Rauschen verwischt den Knick": {**_BASE, "noise": 0.6},
    "Ausreißer ziehen die Achse": {**_BASE, "outlier_pct": 10},
    "Gekrümmte Fläche (PCA scheitert)": {**_BASE, "curvature": 1.0},
    "Vier Faktoren (2 Achsen reichen nicht)": {**_BASE, "q": 4},
}
PRESET_HELP = {
    "Einfaches Beispiel (scharfer Knick)": "Zwei versteckte Faktoren, keine Krümmung, wenig Rauschen: der Scree-Plot hat einen scharfen Knick nach genau zwei Komponenten, "
        "die 2-D-Projektion behält die Nachbarschaft fast perfekt - PCA im Idealfall.",
    "Einheiten-Falle (Rohdaten)": "Dieselben Daten, aber in Einheiten wie gemessen (Meter, Kilogramm, Minuten): die Distanz in Metern hat die größte Zahlenspanne, die erste Komponente "
        "zeigt fast nur auf sie und erklärt 99,98 % der Varianz - ein Artefakt der Einheiten, keine Erkenntnis.",
    "Rauschen verwischt den Knick": "Viel Messrauschen: hinter den zwei echten Komponenten bleibt so viel Varianz übrig, dass PCA 7-8 Komponenten für 90 % braucht - "
        "der Knick im Scree-Plot ist verwischt.",
    "Ausreißer ziehen die Achse": "10 % Sonderfahrten mit stark überhöhtem Zeitdruck: die erste Hauptkomponente dreht sich deutlich gegenüber denselben Touren ohne sie - "
        "Varianz zählt quadratisch, wenige extreme Touren dominieren.",
    "Gekrümmte Fläche (PCA scheitert)": "Die Touren liegen auf einer gebogenen zweidimensionalen Fläche: PCA braucht fünf Komponenten für 90 % (wahre Dimension: 2), die ersten zwei "
        "behalten nur gut die Hälfte der Varianz - die Linearitätsannahme, an der die nächsten Stücke der Linie ansetzen.",
    "Vier Faktoren (2 Achsen reichen nicht)": "Vier echte Faktoren: der Knick liegt sauber bei vier Komponenten, aber die 2-D-Projektion allein verliert einen Teil der Nachbarschaft - "
        "eine Karte in zwei Dimensionen kann nicht alles zeigen.",
}
# Erwartete Messwerte je Preset (mit dem ausgelieferten Code kalibriert; tests/test_presets.py prüft sie). Tupel = (min, max), Text = Verdict-Code.
PRESET_EXPECTED_BANDS = {
    "Einfaches Beispiel (scharfer Knick)": {"verdict": "linear_ok", "k90": (2, 2), "ratio_q": (0.90, 0.96), "trust": (0.98, 1.0)},
    "Einheiten-Falle (Rohdaten)": {"verdict": "units", "k90": (1, 1), "pc1_ratio": (0.999, 1.0), "dominant_share": (0.995, 1.0), "dominant": "Distanz"},
    "Rauschen verwischt den Knick": {"verdict": "noise", "k90": (7, 9), "ratio_q": (0.62, 0.80)},
    "Ausreißer ziehen die Achse": {"verdict": "outliers", "k90": (2, 2), "angle": (10.0, 21.0)},
    "Gekrümmte Fläche (PCA scheitert)": {"verdict": "curvature", "k90": (4, 6), "k90_flat": (2, 2), "ratio_q": (0.45, 0.62), "trust": (0.80, 0.91)},
    "Vier Faktoren (2 Achsen reichen nicht)": {"verdict": "linear_ok", "k90": (4, 4), "trust": (0.72, 0.87)},
}
