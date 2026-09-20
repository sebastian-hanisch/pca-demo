# Hauptkomponentenanalyse (PCA) an Lieferrouten-Kennzahlen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-pca-demo.streamlit.app/)**

Erstes Stück der **Dimensionsreduktion-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research
und Machine Learning": anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein**
Verfahren – die Hauptkomponentenanalyse (PCA) – und lässt stattdessen das **Beispiel wachsen**. Vehikel: 12 Kennzahlen je Lieferroute
(Distanz, Stopps, Ladegewicht, Zeitfenster-Enge, Verspätung, Überstunden, Fahrzeit je km, Stop-and-go-Anteil, Parkzeit, Retourenquote, Sonderwünsche,
Zustellversuche), erzeugt aus **wenigen versteckten Faktoren** - die wahre Dimension q ist damit bekannt und prüfbar.

**Einordnung in die Reihe (die Kanten des Graphen):** PCA ist die **Wurzel** der Dimensionsreduktion-Linie (eigene Sektion, bewusst nicht Teil der
Clustering-Linie). Ihre eigene, ehrlich gezeigte Schwäche ist die **Linearitätsannahme** - genau daran setzen die geplanten Stücke an:
```
pca-demo → Isomap          (geodätische statt gerader Abstände)
pca-demo → LLE             (lokal-lineare Rekonstruktion; Kontrast zu Isomap, kein Fix)
pca-demo → t-SNE → UMAP → PaCMAP
pca-demo → Autoencoder     (lineare Variante = PCA, nichtlineare = die Erweiterung)
```
Noch nicht gebaut; diese Demo ist der Startpunkt.

## Was die Demo zeigt

1. **Projektionsrichtung drehen:** zwei Kennzahlen, eine Gerade, ein Winkel-Regler (mit Abspielen): die Varianz der projizierten Touren als Kurve über dem Winkel -
   Maximum λ₁ bei der ersten Hauptkomponente, Minimum λ₂ genau 90° daneben. Das Kernprinzip "Richtung maximaler Varianz = kleinster Rekonstruktionsfehler".
2. **Alle 12 Kennzahlen:** Scree-Plot (mit der wahren Dimension markiert), 2-D-Projektion (Farbe = versteckter Faktor 1), Ladungen der ersten Komponenten,
   Rekonstruktion einer einzelnen Tour aus *k* Komponenten.
3. **📐 Wie viele Dimensionen stecken wirklich in den Daten?** (live): wahres q gegen k₉₀ (Komponenten für 90 % Varianz), Varianzanteil der ersten q Komponenten,
   Trustworthiness der 2-D-Projektion, dazu ein Sweep über wachsende Krümmung (feste Seeds ab 100000, unabhängig vom Demo-Seed).

Vier Regler führen die Schwierigkeit hoch - jeder zeigt eine bekannte PCA-Falle **gemessen**, nicht behauptet:

| Regler | Falle | Gemessen (Preset, Seed 7) |
|---|---|---|
| **Krümmung** (Hauptschwäche) | gerade Achsen können eine gebogene Fläche nicht mit wenigen Koordinaten beschreiben | q = 2: k₉₀ steigt von 2 auf 5, die ersten 2 Komponenten behalten nur 54 % statt 93 % der Varianz, Trustworthiness 0.86 statt 0.99 |
| **Skalierung** | Rohdaten in Meter/Kilogramm: die größte Zahlenspanne gewinnt | PC1 = 99.98 % der Varianz, 99.9 % davon auf "Distanz [m]"; standardisiert nur 53 % |
| **Ausreißer** | wenige Sonderfahrten ziehen PC1 zu sich (Varianz zählt quadratisch) | 10 % Sonderfahrten drehen PC1 um ~15° gegenüber denselben Touren ohne sie |
| **Rauschen** | verwischt den Knick im Scree-Plot | Rauschen 0.6: k₉₀ = 8 statt 2 |

Über die festen Sweep-Seeds (5 Seeds × 250 Touren, q = 2, Rauschen 0.25, standardisiert) steigt k₉₀ von **2.0** (Krümmung 0) auf **5.0** (Krümmung 1); der Varianzanteil der ersten
zwei Komponenten fällt von 94 % auf 54 %. Eine Trustworthiness-Abnahme setzt erst bei starker Krümmung ein (0.99 → 0.86) - k₉₀ und der Varianzanteil reagieren früher.

## Was PCA nicht kann (und wohin die nächsten Stücke gehen)

- **Nichtlineare Struktur:** siehe oben - die Schwäche, die diese Linie antreibt.
- **Varianz ist nicht Signal:** PCA behält, was viel streut, nicht unbedingt, was für eine Aufgabe zählt.
- **Einheiten und Ausreißer** (oben) - beides lässt sich durch Standardisierung bzw. robuste Varianten mildern, aber nicht durch PCA selbst beheben.

## Modell und Verfahren

- **Generator** (`pca_scenario.py`): latente Faktoren z ~ N(0, I_q); jedes der 12 Merkmale lädt auf den Faktor seiner Gruppe (Größe, Zeitdruck, Verkehr, Sonderfälle;
  bei q < 4 teilen sich mehrere Gruppen einen Faktor) plus kleine Querladungen. Krümmung: `κ·B·h(z)` mit sin/cos, Quadraten und Produkten der Faktoren. Rauschen und
  Sonderfahrten kommen zuletzt hinzu, so dass dieselben Touren mit und ohne Ausreißer vergleichbar sind. Die Matrizen sind fest, nur Touren und Rauschen hängen vom Seed ab.
- **PCA** (`pca_algorithm.py`, ohne sklearn): Zentrieren, optional Standardisieren, Singulärwertzerlegung; deterministische Vorzeichen (größte Ladung positiv).
  Rekonstruktionsfehler bei *k* Komponenten = Summe der verworfenen Eigenwerte (Eckart-Young).
- **Auswertung** (`pca_evaluation.py`): k₉₀, Trustworthiness (Venna & Kaski, eigene Implementierung), Krümmungs-Sweep, Verdict-Kaskade (Einheiten-Falle → Krümmung → Ausreißer → Rauschen → sauberer Fall).

## Verifikation

- PCA gegen `sklearn.decomposition.PCA` (Eigenwerte, Varianzanteile, Komponenten bis auf das Vorzeichen) mit und ohne Standardisierung; Orthonormalität; Varianzen summieren auf die Gesamtvarianz.
- Rekonstruktionsfehler = Summe der verworfenen Eigenwerte (exakt), voller Rang = exakte Rekonstruktion; kein zufälliger Unterraum schlägt PCA (Eckart-Young).
- Winkel-Kurve: Maximum = λ₁ bei PC1, Minimum = λ₂ 90° daneben; Varianz + Rekonstruktionsfehler = Gesamtvarianz für jeden Winkel.
- Standardisierung ist einheitenunabhängig (Merkmale mit Faktoren 1000…0.001 skaliert, gleiches Ergebnis), Rohdaten-PCA nicht.
- Trustworthiness gegen `sklearn.manifold.trustworthiness` (Abweichung < 1e-9).
- Generator: bei Krümmung 0 und kleinem Rauschen genau q signifikante Komponenten (q = 1…4); Krümmung und Rauschen erhöhen k₉₀; gleiche Touren mit/ohne Sonderfahrten.
- Alle 6 Presets in ihren kalibrierten Bändern (`tests/test_presets.py`), AppTest-Rauchtests (Default, jedes Preset, Randgrößen, Zustands-Reset), Achsensperre aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Richtung drehen, Scree/Projektion/Ladungen/Rekonstruktion, 📐 Dimensions-Check, Mathe |
| `pca_algorithm.py` | PCA von Grund auf, Richtungs-Varianzkurve |
| `pca_scenario.py`, `pca_constants.py` | Lieferrouten-Generator, Merkmale, Regler, Presets |
| `pca_evaluation.py` | k₉₀, Trustworthiness, Sweep, Verdict |
| `pca_presets.py` | Permalink, Presets, Zufalls-Seed |
| `pca_visualization.py` | Plotly-Figuren (alle achsengesperrt) |
| `tests/` | PCA-Kreuzvergleiche, Generator, Auswertung, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
