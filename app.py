"""Hauptkomponentenanalyse (PCA) an Lieferrouten-Kennzahlen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - die
Hauptkomponentenanalyse - und lässt stattdessen das Beispiel wachsen: von wenigen versteckten Faktoren, die PCA sauber zurückgewinnt,
bis zu gekrümmten Daten, an denen ihre Linearitätsannahme scheitert. Erstes Stück der Dimensionsreduktion-Linie der "Konzepte"-Reihe
(siehe README für die Einordnung).

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import pca_constants as C
from pca_algorithm import direction, fit_pca, optimal_angle, residual_along_direction, variance_along_direction
from pca_evaluation import analyse, curvature_sweep, verdict
from pca_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from pca_scenario import generate_dataset
from pca_visualization import (
    build_curvature_sweep,
    build_direction_figure,
    build_loadings,
    build_projection,
    build_reconstruction,
    build_scree,
    build_variance_curve,
)

st.set_page_config(page_title="PCA – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _dataset(n_tours, q, curvature, noise, outlier_pct, seed):
    return generate_dataset(n_tours, q, curvature, noise, outlier_pct, seed)


@st.cache_data(show_spinner=False)
def _analysis(n_tours, q, curvature, noise, outlier_pct, seed, standardize):
    dataset = generate_dataset(n_tours, q, curvature, noise, outlier_pct, seed)
    return analyse(dataset, standardize, seed)


@st.cache_data(show_spinner=False)
def _sweep(q, noise, standardize):
    return curvature_sweep(q, noise, standardize)


st.title("🚚 Hauptkomponentenanalyse (PCA) an Lieferrouten-Kennzahlen")
st.markdown(
    """
Für jede Lieferroute liegen **12 Kennzahlen** vor - Distanz, Stopps, Ladegewicht, Zeitfenster-Enge, Verspätung, Überstunden, Fahrzeit je
km und weitere. Sie sind nicht unabhängig: dahinter stehen nur wenige **versteckte Faktoren** (die "Größe" einer Tour, ihr Zeitdruck ...).
Die **Hauptkomponentenanalyse (PCA)** findet die Richtungen im Merkmalsraum, entlang derer die Touren am stärksten variieren, und
komprimiert die 12 Zahlen auf wenige Koordinaten. Genau **wie** das funktioniert, erklärt der aufgeklappte Abschnitt direkt darunter -
bevor weiter unten die Richtung live gedreht wird und die Frage "📐 Wie viele Dimensionen stecken wirklich in den Daten?" live
beantwortet wird.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - erstes Stück der "
    "Dimensionsreduktion-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel: PCA ist linear, und genau an dieser "
    "Annahme scheitert sie, sobald die Daten gekrümmt sind."
)

with st.expander("So funktioniert PCA", expanded=True):
    st.markdown(
        """
PCA sucht eine Richtung im Merkmalsraum, entlang derer die Touren **möglichst stark streuen**, dann - senkrecht dazu - die nächste, und so weiter:

1. **Zentrieren** (und meist **standardisieren**): jedes Merkmal wird auf Mittelwert 0 und - bei z-Werten - Standardabweichung 1 gebracht.
2. **Kovarianz** der Merkmale bestimmen: welche Kennzahlen steigen und fallen gemeinsam?
3. **Hauptkomponenten** = die Eigenvektoren dieser Kovarianzmatrix, sortiert nach ihrem Eigenwert (= die Varianz entlang der Komponente).
   Die erste Komponente ist die Richtung größter Varianz, jede weitere die größte Varianz unter allen Richtungen senkrecht zu den vorigen.
4. **Projizieren**: jede Tour bekommt Koordinaten entlang der ersten *k* Komponenten. Was dabei verloren geht, ist genau die Varianz der
   verworfenen Komponenten - der **Rekonstruktionsfehler**.

Der **Scree-Plot** zeigt, wie viel Varianz jede Komponente trägt. Stecken hinter den Daten wenige echte Faktoren, gibt es einen **scharfen Knick**:
so viele Komponenten wie Faktoren, dahinter nur noch Rauschen. PCA hat aber drei bekannte Schwächen, die diese Demo mit Reglern live zeigt:
**Einheiten** (Rohdaten in Meter und Kilogramm - die größte Einheit gewinnt), **Ausreißer** (wenige Sonderfahrten ziehen die erste Achse zu sich)
und - die Hauptschwäche - **Krümmung**: PCA kennt nur gerade Achsen. Liegen die Touren auf einer gebogenen Fläche, braucht sie viele Komponenten
für etwas, das eigentlich nur wenige Dimensionen hat.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_tours = st.slider("Anzahl Touren", *bounds("n_tours_slider"), key="n_tours_slider", step=50)
    q = st.slider(
        "Wahre Anzahl versteckter Faktoren (q)", *bounds("q_slider"), key="q_slider",
        help="So viele echte Einflussgrößen erzeugen die 12 Kennzahlen. Bei q < 4 teilen sich mehrere Merkmalsgruppen einen Faktor.",
    )
    curvature = st.slider(
        "Krümmung", *bounds("curvature_slider"), key="curvature_slider", step=0.05,
        help="0 = die Kennzahlen hängen linear von den Faktoren ab (PCA-Idealfall). Größer = die Touren liegen auf einer zunehmend gebogenen Fläche.",
    )
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Messrauschen je Kennzahl (in Einheiten der Signalstreuung). Verwischt den Knick im Scree-Plot.",
    )
    outlier_pct = st.slider(
        "Sonderfahrten (Ausreißer, %)", *bounds("outlier_slider"), key="outlier_slider",
        help="Anteil ungewöhnlicher Touren mit stark überhöhtem Zeitdruck-Faktor.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**Vorverarbeitung**")
    scaling = st.radio(
        "Skalierung", options=C.SCALINGS, key="scaling_radio", format_func=lambda s: C.SCALING_LABELS[s],
        help="Rohdaten: Meter, Kilogramm, Minuten wie gemessen - das Merkmal mit der größten Zahlenspanne dominiert. Standardisiert: jedes Merkmal auf Streuung 1.",
    )

    st.button("🎲 Neue Touren generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für die Touren.")

sync_query_params(n_tours, q, curvature, noise, outlier_pct, scaling, seed)

params = (int(n_tours), int(q), float(curvature), float(noise), int(outlier_pct), int(seed))
standardize = scaling == "standardized"
dataset = _dataset(*params)
analysis = _analysis(*params, standardize)
model = analysis.model
data_key = params + (standardize,)

# --- Projektionsrichtung drehen ------------------------------------------------------------------------------------------

st.markdown("## 🎯 Projektionsrichtung drehen")
st.caption(
    "Zwei Kennzahlen, eine Gerade durch die Punktwolke: jede Tour wird senkrecht auf die Gerade projiziert. Gesucht ist die Richtung, "
    "entlang der die projizierten Touren **am stärksten streuen** - gleichbedeutend damit, dass die Abstände zur Geraden (der "
    "Rekonstruktionsfehler) am kleinsten sind. Hier immer in z-Werten, sonst wäre der Winkel eine Frage der Einheiten."
)
fcols = st.columns(2)
default_x, default_y = C.FEATURE_NAMES[0], C.FEATURE_NAMES[2]
if st.session_state.get("feat_x") not in C.FEATURE_NAMES:
    st.session_state["feat_x"] = default_x
if st.session_state.get("feat_y") not in C.FEATURE_NAMES or st.session_state["feat_y"] == st.session_state["feat_x"]:
    st.session_state["feat_y"] = next(f for f in ([default_y] + list(C.FEATURE_NAMES)) if f != st.session_state["feat_x"])
with fcols[0]:
    feat_x = st.selectbox("Merkmal auf der x-Achse", options=C.FEATURE_NAMES, key="feat_x")
with fcols[1]:
    feat_y = st.selectbox("Merkmal auf der y-Achse", options=[f for f in C.FEATURE_NAMES if f != feat_x], key="feat_y")
ix, iy = C.FEATURE_NAMES.index(feat_x), C.FEATURE_NAMES.index(feat_y)
Zall = (dataset.X - dataset.X.mean(0)) / dataset.X.std(0, ddof=1)
Z2 = Zall[:, [ix, iy]]
pc1_angle, lam_max, lam_min = optimal_angle(Z2)

angle_key = data_key + (feat_x, feat_y)
if "angle_slider" not in st.session_state or st.session_state.get("angle_owner") != angle_key:
    st.session_state["angle_slider"] = 0
    st.session_state["angle_owner"] = angle_key


def _jump_to_best():
    st.session_state["angle_slider"] = int(round(pc1_angle)) % 180


angle_col, play_col, jump_col = st.columns([5, 1, 1.4])
with angle_col:
    theta = st.slider("Projektionswinkel (Grad)", 0, 179, key="angle_slider", help="0° = entlang der x-Achse, 90° = entlang der y-Achse.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
with jump_col:
    st.button("🎯 Beste Richtung", width="stretch", on_click=_jump_to_best, help="Springt zum Winkel der ersten Hauptkomponente.")
show_pc1 = st.checkbox("PC1 (beste Richtung) einblenden", value=False, key="show_pc1")

dir_slot = st.empty()
curve_slot = st.empty()


def _render(current_theta):
    dir_slot.plotly_chart(
        build_direction_figure(Z2, current_theta, (feat_x, feat_y), show_pc1, pc1_angle), width="stretch", key=f"direction_{current_theta}"
    )
    curve_slot.plotly_chart(
        build_variance_curve(Z2, current_theta, pc1_angle, lam_max, lam_min), width="stretch", key=f"variance_curve_{current_theta}"
    )


if auto_play:
    for a in range(0, 180, 6):
        _render(a)
        time.sleep(0.15)
    theta = 179
    _render(theta)
else:
    _render(theta)

var_here = variance_along_direction(Z2, theta)
resid_here = residual_along_direction(Z2, theta)
total = lam_max + lam_min
am1, am2, am3, am4 = st.columns(4)
am1.metric("Varianz entlang der Richtung", f"{var_here:.2f}", help=f"Gesamtvarianz beider z-Werte: {total:.2f}.")
am2.metric("Anteil der Gesamtvarianz", f"{var_here / total * 100:.0f} %")
am3.metric(
    "Rekonstruktionsfehler", f"{resid_here:.2f}",
    help="Mittlere quadratische Abweichung der Touren von ihrer Projektion. Varianz + Fehler = Gesamtvarianz (bis auf den Faktor (n−1)/n) - was die Richtung nicht erklärt, geht verloren.",
)
am4.metric(
    "Abstand zum Optimum", f"{lam_max - var_here:.2f}",
    delta=f"{(theta - pc1_angle + 90) % 180 - 90:+.0f}° zur besten Richtung", delta_color="off",
    help="Fehlende Varianz gegenüber dem größten Eigenwert λ₁; bei 0 ist die Richtung gleich PC1.",
)
st.caption(
    f"Beste Richtung (PC1 dieser beiden Merkmale): **{pc1_angle:.0f}°** - dort erklärt die Gerade {lam_max / total * 100:.0f} % der Gesamtvarianz. "
    f"Senkrecht dazu ({(pc1_angle + 90) % 180:.0f}°) liegt das Minimum: die zweite Komponente mit {lam_min / total * 100:.0f} %."
)

st.markdown("---")

# --- Hochdimensional ---------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Alle 12 Kennzahlen: Scree-Plot, Projektion, Ladungen")
sc1, sc2 = st.columns(2)
with sc1:
    st.markdown("**Scree-Plot: wie viel Varianz trägt jede Komponente?**")
    st.plotly_chart(build_scree(analysis.ratio, dataset.q), width="stretch", key="scree")
with sc2:
    st.markdown("**Projektion auf die ersten zwei Komponenten**")
    st.plotly_chart(
        build_projection(model.transform(dataset.X, 2), dataset.z[:, 0], dataset.outlier), width="stretch", key="projection"
    )
st.caption(
    f"Erste Komponente: {analysis.ratio[0] * 100:.1f} % der Varianz, erste zwei: {analysis.ratio[:2].sum() * 100:.1f} %. "
    "Die Farbe ist der (in der Realität unbekannte) versteckte Faktor 1 - verläuft sie glatt über die Projektion, hat PCA ihn zurückgewonnen."
)

st.markdown("**Ladungen: welche Kennzahlen gehören zu welcher Komponente?**")
st.plotly_chart(build_loadings(model.components, C.FEATURE_LABELS), width="stretch", key="loadings")
group_txt = []
for i in range(min(3, dataset.q + 1)):
    share = np.zeros(4)
    for j in range(C.N_FEATURES):
        share[C.GROUP_OF_FEATURE[j]] += model.components[i][j] ** 2
    g = int(share.argmax())
    group_txt.append(f"PC{i + 1}: vor allem **{C.GROUPS[g]}** ({share[g] * 100:.0f} % der Ladung)")
st.caption(
    "Ladungen sind die Einträge der Einheitsvektoren - große Beträge heißen: diese Kennzahl bestimmt die Richtung maßgeblich. "
    + "; ".join(group_txt) + ". Bei q < 4 teilen sich mehrere Gruppen einen Faktor, die Zuordnung ist dann gemischt."
)

st.markdown("**Rekonstruktion: was geht bei k Komponenten verloren?**")
recon_key = data_key
if "recon_k" not in st.session_state or st.session_state.get("recon_owner") != recon_key:
    st.session_state["recon_k"] = int(dataset.q)
    st.session_state["recon_tour"] = 0
    st.session_state["recon_owner"] = recon_key
rk_col, rt_col = st.columns([3, 1])
with rk_col:
    recon_k = st.slider("Anzahl Komponenten k", 1, C.N_FEATURES, key="recon_k")
with rt_col:
    recon_tour = st.number_input("Tour Nr.", 0, int(dataset.n) - 1, key="recon_tour", step=1)
z_original = model.preprocess(dataset.X[int(recon_tour)])
z_recon = (z_original @ model.components[:recon_k].T) @ model.components[:recon_k]
st.plotly_chart(build_reconstruction(z_original, z_recon, C.FEATURE_NAMES, recon_k), width="stretch", key=f"reconstruction_{recon_k}")
lost = float(model.explained_variance[recon_k:].sum() / model.explained_variance.sum() * 100)
rc1, rc2 = st.columns(2)
rc1.metric("Rekonstruktionsfehler (alle Touren)", f"{model.reconstruction_error(dataset.X, recon_k):.3f}",
           help="Mittlere quadratische Abweichung im (skalierten) Merkmalsraum - gleich (n−1)/n mal der Summe der verworfenen Eigenwerte.")
rc2.metric("Verworfene Varianz", f"{lost:.1f} %")

st.markdown("---")

# --- 📐 Wie viele Dimensionen stecken wirklich in den Daten? --------------------------------------------------------------

st.subheader("📐 Wie viele Dimensionen stecken wirklich in den Daten?")
st.markdown(
    f"""
Hier ist die Antwort **bekannt**: die Kennzahlen entstehen aus **q = {dataset.q}** versteckten Faktoren. Ein gutes Dimensionsreduktions-Verfahren sollte
das zurückgewinnen. Live geprüft, nicht behauptet - für Ihr aktuelles Szenario und über feste Sweep-Seeds für wachsende Krümmung:
"""
)
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Wahre Dimension q", dataset.q)
mc2.metric(
    f"PCA: Komponenten für {C.VARIANCE_TARGET * 100:.0f} % Varianz", analysis.k90, delta=f"{analysis.k90 - dataset.q:+d} ggü. q" if analysis.k90 != dataset.q else None,
    delta_color="inverse", help="k₉₀: die kleinste Komponentenzahl, deren kumulierte Varianz 90 % erreicht - PCAs Schätzung der Dimension.",
)
mc3.metric(f"Varianz der ersten {dataset.q} Komponenten", f"{analysis.ratio_q * 100:.0f} %",
           help="Wie viel Varianz PCA behält, wenn man ihr die wahre Dimension verrät.")
mc4.metric("Nachbarschaft erhalten (Trustworthiness, 2-D)", f"{analysis.trust_2d:.2f}",
           help=f"Anteil der Nachbarn in der 2-D-Projektion, die auch im Originalraum Nachbarn sind (k = {C.TRUST_NEIGHBORS}); 1 = perfekt.")

level, code, vdata = verdict(analysis, dataset, standardize)
if code == "units":
    st.warning(
        f"⚠️ **Einheiten-Falle**: in Rohdaten trägt **{vdata['dominant']}** {vdata['dominant_share'] * 100:.1f} % der ersten Komponente, sie erklärt "
        f"{vdata['pc1_ratio'] * 100:.2f} % der Varianz - nicht weil die Distanz so wichtig wäre, sondern weil sie in Metern gemessen wird. Auf "
        "**Standardisiert** umstellen (Regler links), und PCA findet die echten Faktoren."
    )
elif code == "curvature":
    st.warning(
        f"⚠️ **PCA überschätzt die Dimension**: k₉₀ = {vdata['k90']} statt q = {vdata['q']} (ohne Krümmung wären es {vdata['k90_flat']}). Die ersten {vdata['q']} "
        f"Komponenten erklären nur {vdata['ratio_q'] * 100:.0f} % der Varianz. Das ist die Linearitätsannahme: gerade Achsen können eine gebogene Fläche nicht "
        "mit wenigen Koordinaten beschreiben. Verfahren, die der Fläche folgen (Isomap, LLE, t-SNE, Autoencoder), sind die Antwort - Thema der nächsten Stücke."
    )
elif code == "outliers":
    st.warning(
        f"⚠️ **Ausreißer ziehen die Achse**: PC1 dreht sich durch die Sonderfahrten um {vdata['angle']:.0f}° gegenüber denselben Touren ohne sie. "
        "Die Varianz wird quadratisch gezählt - wenige extreme Touren dominieren die Richtung."
    )
elif code == "noise":
    st.warning(
        f"⚠️ **Rauschen verwischt den Knick**: k₉₀ = {vdata['k90']} statt q = {vdata['q']}. Die Rauschdimensionen tragen zusammen so viel Varianz, "
        "dass PCA mehr Komponenten braucht, um 90 % zu erreichen - die echte Struktur steckt in den ersten wenigen, der Scree-Plot zeigt sie noch."
    )
else:
    st.success(
        f"✅ PCA gewinnt die Struktur zurück: k₉₀ = {vdata['k90']} bei q = {vdata['q']}, die ersten {vdata['q']} Komponenten erklären "
        f"{vdata['ratio_q'] * 100:.0f} % der Varianz, die 2-D-Projektion behält die Nachbarschaft (Trustworthiness {vdata['trust']:.2f})."
    )

sweep_rows = _sweep(int(q), float(noise), standardize)
st.markdown("**Und mit wachsender Krümmung?**")
st.caption(
    f"Gleiche Einstellungen (q = {dataset.q}, Rauschen {noise:.2f}, {C.SCALING_LABELS[scaling].split(' (')[0].lower()}), nur die Krümmung wächst; "
    f"Mittel über {len(C.SWEEP_SEEDS)} feste Seeds mit je {C.SWEEP_N_TOURS} Touren (unabhängig vom Demo-Seed)."
)
st.plotly_chart(build_curvature_sweep(sweep_rows, dataset.q, float(curvature)), width="stretch", key="curvature_sweep")
st.caption(
    f"Von Krümmung 0 auf 1 steigt k₉₀ im Mittel von {sweep_rows[0]['k90']:.1f} auf {sweep_rows[-1]['k90']:.1f}, der Varianzanteil der ersten {dataset.q} Komponenten "
    f"fällt von {sweep_rows[0]['ratio_q'] * 100:.0f} % auf {sweep_rows[-1]['ratio_q'] * 100:.0f} %, die Trustworthiness von {sweep_rows[0]['trust']:.2f} auf "
    f"{sweep_rows[-1]['trust']:.2f}. Der Sweep läuft ohne Ausreißer."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Zielfunktion.** Gegeben zentrierte Daten $X \in \mathbb{R}^{n \times d}$ suche eine Richtung $u$ mit $\lVert u \rVert = 1$, die die Varianz der Projektion maximiert:

$$
\max_{\lVert u \rVert = 1} \ \frac{1}{n-1} \lVert X u \rVert^2 \;=\; \max_{\lVert u \rVert = 1} \ u^\top C\, u, \qquad C = \frac{1}{n-1} X^\top X .
$$

Das ist ein **Rayleigh-Quotient**: sein Maximum ist der größte Eigenwert $\lambda_1$ der Kovarianzmatrix $C$, angenommen beim zugehörigen Eigenvektor - der ersten
Hauptkomponente. Die Kurve im ersten Abschnitt ist genau $u(\theta)^\top C\, u(\theta)$: Maximum $\lambda_1$ bei $\theta^*$, Minimum $\lambda_2$ bei $\theta^* + 90^\circ$.
Jede weitere Komponente maximiert dieselbe Größe unter der Nebenbedingung, senkrecht zu den vorigen zu sein.

**Berechnung.** Statt $C$ zu bilden, zerlegt die Demo $X = U \Sigma V^\top$ (Singulärwertzerlegung): die Zeilen von $V^\top$ sind die Hauptkomponenten, $\lambda_i = \sigma_i^2 / (n-1)$.
Numerisch stabiler als die Eigenzerlegung von $C$; die Vorzeichen werden deterministisch festgelegt (größte Ladung positiv).

**Rekonstruktion.** Mit den ersten $k$ Komponenten $V_k$ ist $\hat X = X V_k V_k^\top$. Der mittlere quadratische Fehler ist exakt $(n-1)/n$ mal die Summe der verworfenen Eigenwerte:

$$
\frac{1}{n} \lVert X - \hat X \rVert_F^2 \;=\; \frac{n-1}{n} \sum_{i > k} \lambda_i .
$$

Nach dem Satz von Eckart und Young ist das der kleinstmögliche Fehler unter allen Projektionen auf einen $k$-dimensionalen linearen Unterraum - PCA ist in diesem Sinn optimal,
aber eben nur für **lineare** Unterräume.

**Standardisierung.** PCA auf der Kovarianzmatrix hängt von den Einheiten ab: skaliert man ein Merkmal um den Faktor $c$, wächst seine Varianz um $c^2$ und es dominiert
die erste Komponente. Standardisieren (Streuung 1) macht das Ergebnis einheitenunabhängig - PCA auf der **Korrelationsmatrix**.

**Grenzen.** (1) *Linearität*: gerade Achsen können eine gebogene $q$-dimensionale Fläche nicht mit $q$ Koordinaten beschreiben. (2) *Einheiten*: ohne Standardisierung
gewinnt die größte Zahlenspanne. (3) *Ausreißer*: die Varianz ist ein quadratisches Maß, wenige extreme Punkte lenken die Richtung. (4) *Varianz ist nicht Signal*:
PCA behält, was viel streut - nicht unbedingt, was für eine Aufgabe (Klassentrennung, Vorhersage) wichtig ist.

**Trustworthiness** (Venna & Kaski, 2001): für jede Tour die $k$ nächsten Nachbarn in der Projektion; Nachbarn, die im Originalraum nicht zu den $k$ nächsten gehören,
werden mit ihrem Originalrang über $k$ bestraft: $T = 1 - \frac{2}{nk(2n - 3k - 1)} \sum_i \sum_{j \in U_i} (r(i,j) - k)$; 1 heißt: keine falschen Nachbarn.

Implementiert in `pca_algorithm.py` (PCA, Richtungs-Varianzkurve), `pca_scenario.py` (Lieferrouten-Generator) und `pca_evaluation.py` (k₉₀, Trustworthiness, Sweep, Verdict).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Dimensionsreduktion: von PCA bis Autoencoder](https://sebastianhanisch.net/konzepte-dimensionsreduktion.html)."
)
