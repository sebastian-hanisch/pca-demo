"""Auswertung: wie viele Komponenten braucht PCA, wie gut bleibt die Nachbarschaft erhalten, und woran liegt es (Krümmung, Einheiten,
Ausreißer, Rauschen)? Alle Aussagen werden am Datensatz gemessen - die wahre Dimension q ist bekannt."""

from dataclasses import dataclass

import numpy as np

import pca_constants as C
from pca_algorithm import fit_pca
from pca_scenario import generate_dataset


def k_for_variance(ratio, target=C.VARIANCE_TARGET):
    """Kleinste Komponentenzahl, deren kumulierter Varianzanteil das Ziel erreicht."""
    cumulative = np.cumsum(ratio)
    return int(np.searchsorted(cumulative, target - 1e-12) + 1)


def trustworthiness(X_high, X_low, n_neighbors=C.TRUST_NEIGHBORS):
    """Trustworthiness (Venna & Kaski, 2001): Anteil der Nachbarn im Einbettungsraum, die auch im Originalraum echte Nachbarn sind, mit
    Rang-Strafe für eingeschleppte Fremde. 1 = perfekt. Eigene Implementierung, gegen sklearn geprüft (nur im Test)."""
    n = len(X_high)
    k = n_neighbors
    d_high = np.linalg.norm(X_high[:, None, :] - X_high[None, :, :], axis=-1)
    d_low = np.linalg.norm(X_low[:, None, :] - X_low[None, :, :], axis=-1)
    np.fill_diagonal(d_high, np.inf)
    np.fill_diagonal(d_low, np.inf)
    ranks_high = np.argsort(np.argsort(d_high, axis=1), axis=1) + 1            # Rang 1 = nächster Nachbar
    neighbors_low = np.argsort(d_low, axis=1)[:, :k]
    penalty = 0.0
    for i in range(n):
        r = ranks_high[i, neighbors_low[i]]
        penalty += float(np.maximum(r - k, 0).sum())
    return 1.0 - 2.0 / (n * k * (2 * n - 3 * k - 1)) * penalty


@dataclass(frozen=True)
class Analysis:
    model: object
    ratio: np.ndarray
    k90: int
    k90_flat: int                    # gleiche Einstellungen bei Krümmung 0 (Attribution)
    ratio_q: float                   # Varianzanteil der ersten q Komponenten (q = wahre Dimension)
    trust_2d: float
    dominant_feature: int            # Merkmal mit dem größten Beitrag zu PC1
    dominant_share: float            # dessen Anteil an PC1 (Ladung^2)
    pc1_angle_clean: float           # Winkel (Grad) zwischen PC1 mit und ohne Ausreißer; 0 wenn keine Ausreißer
    pc1_ratio: float


def _pc1_angle(a, b):
    cos = abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))


def analyse(dataset, standardize, seed):
    """Alle Kennzahlen für einen Datensatz (und die Gegenstücke ohne Krümmung / ohne Ausreißer bei gleichem Seed)."""
    model = fit_pca(dataset.X, standardize)
    ratio = model.explained_variance_ratio
    flat = dataset if dataset.curvature == 0 else generate_dataset(dataset.n, dataset.q, 0.0, dataset.noise, dataset.outlier_pct, seed)
    k90_flat = k_for_variance(fit_pca(flat.X, standardize).explained_variance_ratio)
    Z = model.preprocess(dataset.X)
    trust = trustworthiness(Z, model.transform(dataset.X, 2))
    pc1 = model.components[0]
    shares = pc1 ** 2
    angle = 0.0
    if dataset.outlier_pct > 0:
        clean = generate_dataset(dataset.n, dataset.q, dataset.curvature, dataset.noise, 0, seed)
        angle = _pc1_angle(pc1, fit_pca(clean.X, standardize).components[0])
    return Analysis(model=model, ratio=ratio, k90=k_for_variance(ratio), k90_flat=k90_flat, ratio_q=float(ratio[:dataset.q].sum()), trust_2d=trust,
                    dominant_feature=int(shares.argmax()), dominant_share=float(shares.max()), pc1_angle_clean=angle,
                    pc1_ratio=float(ratio[0]))


def verdict(analysis, dataset, standardize):
    """Verdict-Kaskade (Warnungen zuerst) -> (Stufe, Code, Daten)."""
    a = analysis
    data = {"q": dataset.q, "k90": a.k90, "k90_flat": a.k90_flat, "ratio_q": a.ratio_q, "trust": a.trust_2d, "pc1_ratio": a.pc1_ratio,
            "dominant": C.FEATURE_NAMES[a.dominant_feature], "dominant_share": a.dominant_share, "angle": a.pc1_angle_clean}
    if not standardize and a.dominant_share >= C.UNITS_DOMINANCE:
        return "warning", "units", data
    if dataset.curvature > 0 and a.k90 - a.k90_flat >= C.CURVATURE_EXTRA_COMPONENTS:
        return "warning", "curvature", data
    if dataset.outlier_pct > 0 and a.pc1_angle_clean >= C.OUTLIER_ANGLE_DEG:
        return "warning", "outliers", data
    if a.k90_flat - dataset.q >= C.NOISE_EXTRA_COMPONENTS:
        return "warning", "noise", data
    return "success", "linear_ok", data


def curvature_sweep(q, noise, standardize, n_tours=C.SWEEP_N_TOURS, curvatures=C.SWEEP_CURVATURES, seeds=C.SWEEP_SEEDS):
    """Feste Sweep-Seeds (unabhängig vom Demo-Seed): mittleres k_90 und mittlere Trustworthiness (2-D) je Krümmung."""
    rows = []
    for kappa in curvatures:
        k90s, trusts, ratios = [], [], []
        for seed in seeds:
            ds = generate_dataset(n_tours, q, kappa, noise, 0, seed)
            model = fit_pca(ds.X, standardize)
            k90s.append(k_for_variance(model.explained_variance_ratio))
            ratios.append(float(model.explained_variance_ratio[:q].sum()))
            trusts.append(trustworthiness(model.preprocess(ds.X), model.transform(ds.X, 2)))
        rows.append({"curvature": float(kappa), "k90": float(np.mean(k90s)), "trust": float(np.mean(trusts)),
                     "ratio_q": float(np.mean(ratios))})
    return rows
