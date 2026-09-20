"""Plotly-Visualisierungen der PCA-Demo: Projektionsrichtung und Varianzkurve, Scree-Plot, 2-D-Projektion, Ladungen, Rekonstruktion und
der Krümmungs-Sweep. Alle Figuren laufen durch `lock_axes` (Touch-Scrolling-Konvention des Portfolios: keine Zoom-/Pan-Gesten im Chart)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pca_constants as C
from pca_algorithm import direction, residual_along_direction, variance_along_direction

BLUE, ORANGE, GREEN, RED, GRAY = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98"
PC_COLORS = [BLUE, ORANGE, GREEN]


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_direction_figure(Z2, theta_deg, names, show_pc1, pc1_angle, max_projections=40):
    """Punktwolke zweier z-standardisierter Merkmale, die gewählte Projektionsrichtung, die Lotfußpunkte einer Stichprobe und optional PC1."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Z2[:, 0], y=Z2[:, 1], mode="markers", name="Touren",
                             marker=dict(color=BLUE, size=6, opacity=0.55, line=dict(width=0.5, color="white")), hoverinfo="skip"))
    u = direction(theta_deg)
    idx = np.linspace(0, len(Z2) - 1, min(max_projections, len(Z2))).astype(int)
    foot = np.outer(Z2[idx] @ u, u)
    xs, ys = [], []
    for p, f in zip(Z2[idx], foot):
        xs += [p[0], f[0], None]
        ys += [p[1], f[1], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=GRAY, width=1), name="Abweichung (Rekonstruktionsfehler)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=foot[:, 0], y=foot[:, 1], mode="markers", name="Projektion",
                             marker=dict(color=ORANGE, size=6), hoverinfo="skip"))
    reach = float(np.abs(Z2).max()) * 1.05
    fig.add_trace(go.Scatter(x=[-reach * u[0], reach * u[0]], y=[-reach * u[1], reach * u[1]], mode="lines", name=f"Richtung {theta_deg:.0f}°",
                             line=dict(color=ORANGE, width=3), hoverinfo="skip"))
    if show_pc1:
        v = direction(pc1_angle)
        fig.add_trace(go.Scatter(x=[-reach * v[0], reach * v[0]], y=[-reach * v[1], reach * v[1]], mode="lines", name="PC1 (beste Richtung)",
                                 line=dict(color=GREEN, width=2, dash="dash"), hoverinfo="skip"))
    fig.update_xaxes(title=f"{names[0]} (z-Wert)", range=[-reach, reach], zeroline=True, constrain="domain")
    fig.update_yaxes(title=f"{names[1]} (z-Wert)", range=[-reach, reach], scaleanchor="x", scaleratio=1, zeroline=True, constrain="domain")
    fig.update_layout(template="plotly_white", height=430, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_variance_curve(Z2, theta_deg, pc1_angle, lam_max, lam_min):
    """Varianz der Projektion gegen den Winkel: das Maximum liegt bei PC1, das Minimum 90° daneben (Eigenwerte)."""
    angles = np.arange(0, 180.5, 1.0)
    variances = [variance_along_direction(Z2, a) for a in angles]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=angles, y=variances, mode="lines", line=dict(color=BLUE, width=3), name="Varianz der Projektion", hoverinfo="skip"))
    fig.add_hline(y=lam_max, line_dash="dot", line_color=GREEN, annotation_text="λ₁ (größter Eigenwert)", annotation_position="top left")
    fig.add_hline(y=lam_min, line_dash="dot", line_color=RED, annotation_text="λ₂ (kleinster)", annotation_position="bottom left")
    fig.add_trace(go.Scatter(x=[theta_deg], y=[variance_along_direction(Z2, theta_deg)], mode="markers", name="gewählte Richtung",
                             marker=dict(color=ORANGE, size=14, line=dict(width=1.5, color="#14233B")), hoverinfo="skip"))
    fig.add_vline(x=pc1_angle, line_dash="dash", line_color=GREEN)
    fig.update_xaxes(title="Projektionswinkel (Grad)", range=[0, 180], dtick=30)
    fig.update_yaxes(title="Varianz entlang der Richtung", rangemode="tozero")
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return lock_axes(fig)


def build_scree(ratio, q, target=C.VARIANCE_TARGET):
    """Scree-Plot: Varianzanteil je Komponente (Balken) und kumuliert (Linie); die wahre Dimension q ist markiert."""
    d = len(ratio)
    comps = np.arange(1, d + 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=comps, y=ratio * 100, name="je Komponente", marker_color=[BLUE if i <= q else "#9db8d2" for i in comps],
                         hovertemplate="PC%{x}: %{y:.1f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=comps, y=np.cumsum(ratio) * 100, mode="lines+markers", name="kumuliert", line=dict(color=ORANGE, width=3),
                             hovertemplate="bis PC%{x}: %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=target * 100, line_dash="dot", line_color=GRAY, annotation_text=f"{target * 100:.0f} %", annotation_position="bottom right")
    fig.add_vline(x=q + 0.5, line_dash="dash", line_color=GREEN, annotation_text=f"wahre Dimension q = {q}", annotation_position="top right")
    fig.update_xaxes(title="Hauptkomponente", dtick=1)
    fig.update_yaxes(title="Erklärte Varianz (%)", range=[0, 105])
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_projection(scores, color_values, outlier, label="latenter Faktor 1"):
    """Erste zwei Hauptkomponenten; Farbe = latenter Faktor 1 (Wahrheit). Sonderfahrten sind als Kreuz markiert."""
    fig = go.Figure()
    normal = ~outlier
    fig.add_trace(go.Scatter(x=scores[normal, 0], y=scores[normal, 1], mode="markers", name="Touren",
                             marker=dict(color=color_values[normal], colorscale="Viridis", size=7, showscale=True,
                                         colorbar=dict(title=label), line=dict(width=0.5, color="white")), hoverinfo="skip"))
    if outlier.any():
        fig.add_trace(go.Scatter(x=scores[outlier, 0], y=scores[outlier, 1], mode="markers", name="Sonderfahrten",
                                 marker=dict(color=RED, size=9, symbol="x"), hoverinfo="skip"))
    fig.update_xaxes(title="PC1")
    fig.update_yaxes(title="PC2")
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_loadings(components, names, k_show=3):
    """Ladungen der ersten Hauptkomponenten je Merkmal (die Einträge der Einheitsvektoren)."""
    fig = go.Figure()
    for i in range(k_show):
        fig.add_trace(go.Bar(x=list(names), y=components[i], name=f"PC{i + 1}", marker_color=PC_COLORS[i % len(PC_COLORS)]))
    fig.update_layout(barmode="group", template="plotly_white", height=340, margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=-0.35))
    fig.update_yaxes(title="Ladung")
    fig.update_xaxes(tickangle=-40)
    return lock_axes(fig)


def build_reconstruction(z_original, z_reconstructed, names, k):
    """Eine Tour: z-Werte der 12 Merkmale, original gegen rekonstruiert aus den ersten k Komponenten."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(names), y=z_original, name="Original", marker_color=BLUE))
    fig.add_trace(go.Bar(x=list(names), y=z_reconstructed, name=f"aus {k} Komponente{'n' if k != 1 else ''}", marker_color=ORANGE))
    fig.update_layout(barmode="group", template="plotly_white", height=320, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.35))
    fig.update_yaxes(title="Merkmalswert (z-Wert)")
    fig.update_xaxes(tickangle=-40)
    return lock_axes(fig)


def build_curvature_sweep(rows, q, current_curvature):
    """Krümmungs-Sweep über feste Seeds: links k_90 gegen die Krümmung, rechts der Varianzanteil der ersten q Komponenten und die Trustworthiness."""
    x = [r["curvature"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Komponenten für 90 % Varianz (k₉₀)", "Was PCA behält"))
    fig.add_trace(go.Scatter(x=x, y=[r["k90"] for r in rows], mode="lines+markers", name="k₉₀", line=dict(color=ORANGE, width=3)), row=1, col=1)
    fig.add_hline(y=q, line_dash="dash", line_color=GREEN, annotation_text=f"wahre Dimension q = {q}", annotation_position="top left", row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[r["ratio_q"] for r in rows], mode="lines+markers", name=f"Varianz der ersten {q} Komp.",
                             line=dict(color=BLUE, width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=[r["trust"] for r in rows], mode="lines+markers", name="Trustworthiness (2-D)",
                             line=dict(color=RED, width=3)), row=1, col=2)
    for col in (1, 2):
        fig.add_vline(x=current_curvature, line_dash="dot", line_color=GRAY, row=1, col=col)
    fig.update_xaxes(title_text="Krümmung κ")
    fig.update_yaxes(rangemode="tozero", row=1, col=1)
    fig.update_yaxes(range=[0, 1.02], row=1, col=2)
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


__all__ = ["build_direction_figure", "build_variance_curve", "build_scree", "build_projection", "build_loadings", "build_reconstruction",
           "build_curvature_sweep", "lock_axes", "residual_along_direction"]
