#!/usr/bin/env python3
"""Generate eval_results/mini_eval/oellm_eval_results.png for the README."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MINI_EVAL = REPO_ROOT / "eval_results" / "mini_eval"
FULL_EVAL = REPO_ROOT / "eval_results" / "full_eval"
OUT_PATH = MINI_EVAL / "oellm_eval_results.png"

# model_key -> (label, csv_path)
MODELS = {
    "qwen2-0.5b":       ("Qwen2 0.5B (easy)",          MINI_EVAL / "qwen2-0.5b" / "summary.csv"),
    "qwen2-1.5b":       ("Qwen2 1.5B (easy)",          MINI_EVAL / "qwen2-1.5b" / "summary.csv"),
    "gemma4-easy":      ("Gemma 4 E4B (easy)",         MINI_EVAL / "gemma4" / "summary.csv"),
    "gemma3-4b":        ("Gemma 3 4B (harder)",        FULL_EVAL / "gemma3-4b" / "summary.csv"),
    "gemma4-e2b":       ("Gemma 4 E2B (harder)",       FULL_EVAL / "gemma4-e2b" / "summary.csv"),
    "gemma4-harder":    ("Gemma 4 E4B (harder)",       FULL_EVAL / "gemma4" / "summary.csv"),
}
MODEL_COLORS = {
    "qwen2-0.5b":    "#7EC8E3",
    "qwen2-1.5b":    "#0E86D4",
    "gemma4-easy":   "#FF6B6B",
    "gemma3-4b":     "#A8D8A8",
    "gemma4-e2b":    "#FFD700",
    "gemma4-harder": "#FF9900",
}

EU_OFFICIAL = {"bg","hr","cs","da","nl","et","fi","fr","de","el","hu","ga",
               "it","lv","lt","mt","pl","pt","ro","sk","sl","es","sv","en"}

LANG_NAMES = {
    "bg":"Bulgarian","hr":"Croatian","cs":"Czech","da":"Danish","nl":"Dutch",
    "et":"Estonian","fi":"Finnish","fr":"French","de":"German","el":"Greek",
    "hu":"Hungarian","ga":"Irish","it":"Italian","lv":"Latvian","lt":"Lithuanian",
    "mt":"Maltese","pl":"Polish","pt":"Portuguese","ro":"Romanian","sk":"Slovak",
    "sl":"Slovenian","es":"Spanish","sv":"Swedish","en":"English",
    "sq":"Albanian","eu":"Basque","bs":"Bosnian","ca":"Catalan","gl":"Galician",
    "is":"Icelandic","lb":"Luxembourgish","mk":"Macedonian","no":"Norwegian",
    "ru":"Russian","sr":"Serbian","tr":"Turkish","uk":"Ukrainian","cy":"Welsh",
}


def load_scores(csv_path: Path) -> dict[str, float]:
    scores = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores[row["lang"]] = float(row["accuracy"])
    return scores


def main() -> None:
    all_scores = {k: load_scores(v[1]) for k, v in MODELS.items() if v[1].exists()}

    # Sort: extra langs at bottom (low y), EU official at top (high y)
    # Within each group sort ascending so highest scorers end up at the very top
    gemma4 = all_scores.get("gemma4-harder", all_scores.get("gemma4-easy", {}))
    eu_langs = sorted(EU_OFFICIAL, key=lambda l: gemma4.get(l, 0))
    extra_langs = sorted(set(LANG_NAMES) - EU_OFFICIAL, key=lambda l: gemma4.get(l, 0))
    langs = extra_langs + eu_langs  # EU official gets higher y_positions → top of chart

    n_langs = len(langs)
    active_models = [(k, v) for k, v in MODELS.items() if k in all_scores]
    n_models = len(active_models)
    bar_h = 0.20
    group_gap = 0.18
    y_positions = np.arange(n_langs) * (n_models * bar_h + group_gap)

    fig, ax = plt.subplots(figsize=(13, 16))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    for i, (model_key, (model_label, _)) in enumerate(active_models):
        scores = all_scores[model_key]
        color = MODEL_COLORS[model_key]
        offset = (i - (n_models - 1) / 2) * bar_h
        ys = y_positions + offset
        xs = [scores.get(l, 0) for l in langs]

        ax.barh(ys, xs, height=bar_h * 0.85, color=color,
                alpha=0.90, label=model_label, zorder=3)

        for y, x in zip(ys, xs):
            if x >= 0.12:
                ax.text(x - 0.012, y, f"{x:.0%}", ha="right", va="center",
                        fontsize=6.5, color="white", fontweight="bold", zorder=4)

    # Y-axis labels — color EU official slightly brighter
    ax.set_yticks(y_positions)
    y_labels = []
    for l in langs:
        if l in EU_OFFICIAL:
            y_labels.append(f"  {LANG_NAMES[l]}  ({l})")
        else:
            y_labels.append(f"  {LANG_NAMES[l]}  ({l})")
    ax.set_yticklabels(y_labels, fontsize=8.8, color="#D0D0D0")
    ax.tick_params(axis="y", length=0)

    # Color EU official labels gold, extra European blue
    # langs order: extra_langs[0..13] → low y (bottom), eu_langs[14..37] → high y (top)
    # matplotlib renders yticks bottom-to-top, matching the langs order
    for tick, l in zip(ax.get_yticklabels(), langs):
        tick.set_color("#F0C060" if l in EU_OFFICIAL else "#9EB8D0")

    # Separator between extra (bottom) and EU official (top)
    sep_y = (y_positions[len(extra_langs) - 1] + y_positions[len(extra_langs)]) / 2
    ax.axhline(sep_y, color="#404050", linewidth=1.0, linestyle="--", zorder=2)

    # Section labels on the right
    eu_mid_y = np.mean(y_positions[len(extra_langs):])
    extra_mid_y = np.mean(y_positions[:len(extra_langs)])
    ax.text(1.005, eu_mid_y, "EU\nOfficial", transform=ax.get_yaxis_transform(),
            fontsize=7.5, color="#F0C060", va="center", ha="left", fontstyle="italic")
    ax.text(1.005, extra_mid_y, "Additional\nEuropean", transform=ax.get_yaxis_transform(),
            fontsize=7.5, color="#9EB8D0", va="center", ha="left", fontstyle="italic")

    # X-axis
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                       fontsize=9, color="#808080")
    ax.tick_params(axis="x", colors="#404040", top=True, bottom=True)

    # Vertical reference lines
    for x in [0.25, 0.5, 0.75, 1.0]:
        ax.axvline(x, color="#1E2030", linewidth=0.8, zorder=0)

    # Grid
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title
    ax.set_title("OELLM Eval — NIAH Accuracy across 38 Languages",
                 fontsize=13.5, fontweight="bold", color="white", pad=14)
    ax.set_xlabel("Accuracy  (2 questions per language · easy=noun-only context · harder=real book + distractors)",
                  fontsize=8.5, color="#707070", labelpad=10)

    # Legend with averages
    legend_patches = []
    for model_key, (model_label, _) in active_models:
        scores = all_scores[model_key]
        avg = sum(scores.get(l, 0) for l in langs) / len(langs)
        label = f"{model_label}   (avg {avg:.0%})"
        legend_patches.append(mpatches.Patch(color=MODEL_COLORS[model_key], label=label, alpha=0.9))
    leg = ax.legend(handles=legend_patches, loc="lower right",
                    framealpha=0.2, labelcolor="white", fontsize=9.5,
                    edgecolor="#333", facecolor="#1A1A2A")
    leg.get_frame().set_linewidth(0.5)

    # Footer note
    ax.text(0.0, 1.005,
            "★ harder eval: 2000-word context, distracting 7-digit numbers, varied needle depth 10–90%",
            transform=ax.transAxes, fontsize=7, color="#606060", va="bottom")

    plt.tight_layout(rect=[0, 0, 0.97, 1])
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
