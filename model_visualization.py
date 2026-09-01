"""Reusable plots for exploratory HateCR model predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROFILE_LABELS = {
    "balanced": "Balanceado",
    "high_precision": "Alta precisión",
}
PROFILE_COLORS = {
    "balanced": "#C44A32",
    "high_precision": "#315C6B",
}


def _binary_series(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        raise KeyError(f"Missing prediction column: {column}")
    values = pd.to_numeric(data[column], errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError(f"Prediction column must be complete and binary: {column}")
    return values.astype(int)


def summarize_prediction_profiles(
    data: pd.DataFrame,
    profile_columns: Mapping[str, str],
) -> pd.DataFrame:
    rows = []
    for profile, column in profile_columns.items():
        predictions = _binary_series(data, column)
        positive = int(predictions.sum())
        rows.append(
            {
                "profile": profile,
                "profile_label": PROFILE_LABELS.get(profile, profile),
                "prediction_column": column,
                "n_rows": int(len(predictions)),
                "predicted_positive": positive,
                "predicted_positive_pct": 100 * positive / len(predictions),
            }
        )
    return pd.DataFrame(rows)


def summarize_profiles_by_group(
    data: pd.DataFrame,
    group_column: str,
    profile_columns: Mapping[str, str],
) -> pd.DataFrame:
    if group_column not in data.columns:
        raise KeyError(f"Missing group column: {group_column}")
    working = data.copy()
    working[group_column] = working[group_column].fillna("[sin dato]").astype(str)
    rows = []
    for group_value, group in working.groupby(group_column, dropna=False):
        for profile, column in profile_columns.items():
            predictions = _binary_series(group, column)
            positive = int(predictions.sum())
            rows.append(
                {
                    group_column: group_value,
                    "profile": profile,
                    "profile_label": PROFILE_LABELS.get(profile, profile),
                    "n_rows": int(len(group)),
                    "predicted_positive": positive,
                    "predicted_positive_pct": 100 * positive / len(group),
                }
            )
    return pd.DataFrame(rows)


def summarize_hostility_hate_overlap(
    data: pd.DataFrame,
    hostility_column: str,
    hate_column: str,
) -> pd.DataFrame:
    hostility = _binary_series(data, hostility_column)
    hate = _binary_series(data, hate_column)
    categories = np.select(
        [
            hostility.eq(0) & hate.eq(0),
            hostility.eq(1) & hate.eq(0),
            hostility.eq(0) & hate.eq(1),
            hostility.eq(1) & hate.eq(1),
        ],
        [
            "Sin hostilidad ni odio",
            "Solo hostilidad",
            "Solo odio",
            "Hostilidad y odio",
        ],
        default="Inconsistente",
    )
    counts = pd.Series(categories).value_counts()
    order = [
        "Sin hostilidad ni odio",
        "Solo hostilidad",
        "Solo odio",
        "Hostilidad y odio",
    ]
    result = counts.reindex(order, fill_value=0).rename_axis("category").reset_index(
        name="n_rows"
    )
    result["percentage"] = 100 * result["n_rows"] / len(data)
    return result


def save_overall_profile_figure(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_data = summary.copy()
    colors = [PROFILE_COLORS.get(value, "#777777") for value in plot_data["profile"]]
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    bars = ax.bar(
        plot_data["profile_label"],
        plot_data["predicted_positive_pct"],
        color=colors,
        width=0.56,
    )
    maximum = float(plot_data["predicted_positive_pct"].max())
    ax.set_ylim(0, max(4.2, maximum * 1.5))
    ax.set_ylabel("Comentarios predichos como odio (%)")
    ax.set_title("Predicción exploratoria de odio en el corpus formal")
    ax.grid(axis="y", alpha=0.2)
    for bar, row in zip(bars, plot_data.itertuples(index=False)):
        percentage_label = f"{row.predicted_positive_pct:.2f}".replace(".", ",")
        positive_label = f"{int(row.predicted_positive):,}".replace(",", ".")
        total_label = f"{int(row.n_rows):,}".replace(",", ".")
        profile_note = (
            "Mayor cobertura"
            if row.profile == "balanced"
            else "Criterio conservador"
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(0.08, maximum * 0.04),
            (
                f"{percentage_label}%\n"
                f"{positive_label} de {total_label} comentarios\n"
                f"{profile_note}"
            ),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    if len(plot_data) == 2:
        balanced = plot_data[plot_data["profile"].eq("balanced")].iloc[0]
        conservative = plot_data[
            plot_data["profile"].eq("high_precision")
        ].iloc[0]
        difference_n = int(
            balanced["predicted_positive"]
            - conservative["predicted_positive"]
        )
        difference_pct = float(
            balanced["predicted_positive_pct"]
            - conservative["predicted_positive_pct"]
        )
        difference_label = f"{difference_pct:.2f}".replace(".", ",")
        ax.text(
            0.5,
            0.91,
            (
                "Alta precisión selecciona "
                f"{difference_n} comentarios menos "
                f"(−{difference_label} puntos porcentuales)"
            ),
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#F3EFE7",
                "edgecolor": "#CFC6B8",
            },
        )
    fig.text(
        0.5,
        0.045,
        (
            "El perfil de alta precisión reduce falsos positivos, pero deja "
            "escapar más casos potenciales."
        ),
        ha="center",
        color="#333333",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.012,
        "Predicciones del modelo v2; no representan prevalencia confirmada.",
        ha="center",
        color="#555555",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_score_distribution_figure(
    scores: pd.Series,
    thresholds: Mapping[str, float],
    output_path: Path,
) -> None:
    values = pd.to_numeric(scores, errors="coerce").dropna().astype(float)
    if values.empty or not values.between(0, 1).all():
        raise ValueError("Hate score must contain probabilities between zero and one")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    weights = np.full(len(values), 100 / len(values))
    ax.hist(values, bins=np.linspace(0, 1, 31), weights=weights, color="#D6A15D")
    for profile, threshold in thresholds.items():
        ax.axvline(
            threshold,
            color=PROFILE_COLORS.get(profile, "#555555"),
            linewidth=2.2,
            label=(
                f"{PROFILE_LABELS.get(profile, profile)}: "
                f"{threshold:.3f}"
            ),
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilidad estimada de odio")
    ax.set_ylabel("Porcentaje del corpus por intervalo")
    ax.set_title("Distribución del score de odio y umbrales operativos")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_group_profile_figure(
    summary: pd.DataFrame,
    group_column: str,
    output_path: Path,
    title: str,
    min_rows: int = 1,
    max_groups: Optional[int] = None,
) -> pd.DataFrame:
    eligible = summary[summary["n_rows"].ge(min_rows)].copy()
    balanced = eligible[eligible["profile"].eq("balanced")].sort_values(
        ["predicted_positive_pct", "n_rows"], ascending=[True, True]
    )
    if max_groups is not None:
        balanced = balanced.tail(max_groups)
    groups = balanced[group_column].tolist()
    plot_data = eligible[eligible[group_column].isin(groups)].copy()
    y = np.arange(len(groups))
    height = 0.34
    fig_height = max(5.5, 0.58 * len(groups) + 2.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    maximum = float(plot_data["predicted_positive_pct"].max()) if len(plot_data) else 0
    for offset, profile in zip([-height / 2, height / 2], profile_columns_order()):
        subset = plot_data[plot_data["profile"].eq(profile)].set_index(group_column)
        rates = subset.reindex(groups)["predicted_positive_pct"].fillna(0).to_numpy()
        counts = subset.reindex(groups)["predicted_positive"].fillna(0).astype(int).to_numpy()
        bars = ax.barh(
            y + offset,
            rates,
            height,
            label=PROFILE_LABELS.get(profile, profile),
            color=PROFILE_COLORS.get(profile, "#777777"),
        )
        for bar, rate, count in zip(bars, rates, counts):
            ax.text(
                rate + max(0.08, maximum * 0.012),
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}% (n={count})",
                va="center",
                fontsize=8.5,
            )
    group_sizes = (
        plot_data.groupby(group_column)["n_rows"].first().reindex(groups).astype(int)
    )
    ax.set_yticks(y, [f"{group} (N={group_sizes[group]:,})" for group in groups])
    ax.set_xlim(0, max(5.0, maximum * 1.35))
    ax.set_xlabel("Comentarios predichos como odio (%)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    if min_rows > 1:
        fig.text(
            0.5,
            0.01,
            f"Solo grupos con al menos {min_rows} comentarios; tasas exploratorias.",
            ha="center",
            color="#555555",
            fontsize=9,
        )
    fig.tight_layout(rect=[0, 0.04 if min_rows > 1 else 0, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_data


def profile_columns_order() -> Sequence[str]:
    return ("balanced", "high_precision")


def save_overlap_figure(summary: pd.DataFrame, output_path: Path) -> None:
    colors = ["#D5D1C7", "#D58B4A", "#5D7692", "#A53D33"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    bars = ax.barh(summary["category"], summary["percentage"], color=colors)
    maximum = float(summary["percentage"].max())
    ax.set_xlim(0, max(5.0, maximum * 1.22))
    ax.set_xlabel("Porcentaje del corpus")
    ax.set_title("Solapamiento de predicciones balanceadas: hostilidad y odio")
    ax.grid(axis="x", alpha=0.2)
    for bar, row in zip(bars, summary.itertuples(index=False)):
        ax.text(
            bar.get_width() + max(0.08, maximum * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{row.n_rows:,} ({row.percentage:.2f}%)",
            va="center",
            fontweight="bold",
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_initial_reference_metrics_figure(
    reference_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the metrics originally reported for the first labeled sample."""
    metric_columns = [
        "precision_positive",
        "recall_positive",
        "f1_positive",
    ]
    metric_labels = {
        "precision_positive": "Precisión",
        "recall_positive": "Recall",
        "f1_positive": "F1",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True)
    for ax, (target, target_label) in zip(
        axes, [("hostility", "Hostilidad"), ("hate", "Odio")]
    ):
        subset = reference_df[reference_df["target"].eq(target)].copy()
        x = np.arange(len(subset))
        width = 0.24
        for offset, metric in enumerate(metric_columns):
            ax.bar(
                x + (offset - 1) * width,
                subset[metric],
                width,
                label=metric_labels[metric],
            )
        ax.set_xticks(x, subset["model_label"], rotation=18, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Métrica")
        ax.set_title(target_label)
        ax.grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Primer baseline: métricas originalmente reportadas", y=1.02)
    fig.text(
        0.5,
        0.01,
        "Supervisados: holdout n=45. Lexicón: muestra manual completa n=180.",
        ha="center",
        color="#555555",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_initial_variability_figure(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot F1 mean, standard deviation, and range across repeated CV folds."""
    model_labels = {
        "lexicon_hits": "Lexicón",
        "logreg_tfidf": "Logistic Regression",
        "linearsvc_tfidf": "LinearSVC",
    }
    model_colors = {
        "lexicon_hits": "#9A8F76",
        "logreg_tfidf": "#4F758B",
        "linearsvc_tfidf": "#D08C60",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharex=True)
    for ax, (target, target_label) in zip(
        axes, [("hostility", "Hostilidad"), ("hate", "Odio")]
    ):
        subset = summary_df[summary_df["target"].eq(target)].copy()
        subset = subset.sort_values("f1_positive_mean")
        y = np.arange(len(subset))
        for position, row in enumerate(subset.itertuples(index=False)):
            color = model_colors.get(row.model, "#777777")
            ax.hlines(
                position,
                row.f1_positive_min,
                row.f1_positive_max,
                color=color,
                linewidth=1.4,
                alpha=0.55,
            )
            ax.errorbar(
                row.f1_positive_mean,
                position,
                xerr=row.f1_positive_std,
                fmt="o",
                color=color,
                capsize=5,
                linewidth=2.2,
                markersize=7,
            )
            ax.text(
                min(0.98, row.f1_positive_max + 0.025),
                position,
                f"{row.f1_positive_mean:.3f} ± {row.f1_positive_std:.3f}",
                va="center",
                fontsize=9,
            )
        ax.set_yticks(y, [model_labels.get(value, value) for value in subset["model"]])
        ax.set_xlim(0, 1)
        ax.set_xlabel("F1 de la clase positiva")
        ax.set_title(target_label)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Variabilidad del primer baseline en 25 evaluaciones", y=1.02)
    fig.text(
        0.5,
        0.01,
        "Punto: media; barras con topes: ± desviación estándar; línea fina: mínimo–máximo.",
        ha="center",
        color="#555555",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
