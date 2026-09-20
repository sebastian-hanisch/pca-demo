"""PCA von Grund auf (ohne sklearn): Zentrieren, optional Standardisieren, Singulärwertzerlegung.

Für die zentrierte (und ggf. standardisierte) Datenmatrix X_c [n, d] ist X_c = U S V^T. Die Zeilen von V^T sind die Hauptkomponenten
(Eigenvektoren der Kovarianzmatrix), die Eigenwerte sind lambda_i = s_i^2 / (n - 1). Die Vorzeichen sind deterministisch festgelegt
(größte Ladung jeder Komponente positiv), damit Ergebnisse reproduzierbar und vergleichbar sind."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAModel:
    mean: np.ndarray               # [d]
    scale: np.ndarray              # [d] (1 bei Rohdaten-PCA)
    components: np.ndarray         # [d, d]: Zeile i = i-te Hauptkomponente (Einheitsvektor im (skalierten) Merkmalsraum)
    explained_variance: np.ndarray  # [d] Eigenwerte lambda_i, absteigend
    standardized: bool

    @property
    def explained_variance_ratio(self):
        return self.explained_variance / self.explained_variance.sum()

    @property
    def d(self):
        return len(self.mean)

    def preprocess(self, X):
        return (np.asarray(X) - self.mean) / self.scale

    def transform(self, X, k=None):
        """Koordinaten in den ersten k Hauptkomponenten. -> [n, k]"""
        k = self.d if k is None else k
        return self.preprocess(X) @ self.components[:k].T

    def inverse_transform(self, scores):
        """Zurück in den Originalraum aus den Koordinaten der ersten k Komponenten (k = scores.shape[1])."""
        k = scores.shape[1]
        return (scores @ self.components[:k]) * self.scale + self.mean

    def reconstruct(self, X, k):
        return self.inverse_transform(self.transform(X, k))

    def reconstruction_error(self, X, k):
        """Mittlere quadratische Rekonstruktionsabweichung je Tour im (skalierten) Raum. Für k Komponenten gleich der Summe der
        verworfenen Eigenwerte (mal (n-1)/n)."""
        Z = self.preprocess(X)
        residual = Z - (Z @ self.components[:k].T) @ self.components[:k]
        return float((residual ** 2).sum(1).mean())


def _fix_signs(components):
    signs = np.sign(components[np.arange(len(components)), np.abs(components).argmax(1)])
    signs[signs == 0] = 1.0
    return components * signs[:, None]


def fit_pca(X, standardize):
    X = np.asarray(X, dtype=float)
    mean = X.mean(0)
    scale = X.std(0, ddof=1) if standardize else np.ones(X.shape[1])
    scale = np.where(scale > 0, scale, 1.0)
    Z = (X - mean) / scale
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    variance = s ** 2 / (len(X) - 1)
    return PCAModel(mean=mean, scale=scale, components=_fix_signs(vt), explained_variance=variance, standardized=standardize)


# --- Abschnitt "Projektionsrichtung drehen" (zwei Merkmale) ---------------------------------------------------------

def direction(theta_deg):
    t = np.radians(theta_deg)
    return np.array([np.cos(t), np.sin(t)])


def variance_along_direction(Z2, theta_deg):
    """Varianz der auf die Richtung theta projizierten Punkte (Z2 [n, 2], zentriert): der Rayleigh-Quotient u^T C u."""
    u = direction(theta_deg)
    return float(np.var(Z2 @ u, ddof=1))


def residual_along_direction(Z2, theta_deg):
    """Mittlere quadratische Abweichung der Punkte von ihrer Projektion auf die Gerade (= der Rekonstruktionsfehler bei k = 1)."""
    u = direction(theta_deg)
    along = Z2 @ u
    return float(((Z2 - np.outer(along, u)) ** 2).sum(1).mean())


def optimal_angle(Z2):
    """Winkel (Grad, in [0, 180)) der ersten Hauptkomponente von Z2 und die beiden Eigenwerte (groß, klein)."""
    model = fit_pca(Z2, standardize=False)
    v = model.components[0]
    return float(np.degrees(np.arctan2(v[1], v[0])) % 180.0), float(model.explained_variance[0]), float(model.explained_variance[1])
