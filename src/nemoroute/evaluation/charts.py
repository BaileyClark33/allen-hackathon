from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nemoroute.evaluation.system import StrategySummary

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
SERIES = ["#2563eb", "#d97706", "#059669", "#7c3aed", "#dc2626"]


def _style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def confusion_chart(matrix: np.ndarray, tiers: list[int], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(tiers)), [f"Tier {t}" for t in tiers])
    ax.set_yticks(range(len(tiers)), [f"Tier {t}" for t in tiers])
    ax.set_xlabel("Predicted tier", color=INK)
    ax.set_ylabel("Label (cheapest passing tier)", color=INK)
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else INK, fontsize=11)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cost_quality_chart(
    summaries: list[StrategySummary], path: Path, y_label: str = "Mean judge score (1-5)"
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    _style(ax)
    for i, s in enumerate(summaries):
        color = SERIES[i % len(SERIES)]
        cost = s.mean_cost_usd * 1000
        xerr = [[cost - s.cost_ci[0] * 1000], [s.cost_ci[1] * 1000 - cost]]
        yerr = [[s.mean_score - s.score_ci[0]], [s.score_ci[1] - s.mean_score]]
        ax.errorbar(cost, s.mean_score, xerr=xerr, yerr=yerr, fmt="o", color=color,
                    ecolor=color, elinewidth=1, capsize=3, markersize=7)
        ax.annotate(s.strategy, (cost, s.mean_score), textcoords="offset points",
                    xytext=(7, 6), color=INK, fontsize=9)
    ax.set_xlabel("Mean cost per query (USD x 1,000)", color=INK)
    ax.set_ylabel(y_label, color=INK)
    ax.set_title("Cost vs. quality (95% bootstrap CIs)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def latency_chart(summaries: list[StrategySummary], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    _style(ax)
    names = [s.strategy for s in summaries]
    x = np.arange(len(names))
    width = 0.38
    ax.bar(x - width / 2, [s.p50_latency_s for s in summaries], width, label="p50", color=SERIES[0])
    ax.bar(x + width / 2, [s.p95_latency_s for s in summaries], width, label="p95", color=SERIES[1])
    ax.set_xticks(x, names)
    ax.set_ylabel("Latency (s)", color=INK)
    ax.set_title("End-to-end latency per query", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
