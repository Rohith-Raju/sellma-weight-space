"""Regenerate every figure for the report from the saved result files.

Run from the project root:
    uv run --with matplotlib python -I report/figures/make_figures.py

Reads only gpu_results/experiments/analysis/ (or results/experiments/analysis/ when gpu_results/ does
not exist, as in a fresh clone) and writes fig*.png (200 dpi) and fig*.svg next to
this script. Before drawing, it recomputes the key numbers and compares them with the verified values
in verification/CLAIMS.md (with the corrections in verification/SUMMARY.md); any mismatch is printed.

Colours, marks and chrome follow the dataviz skill's reference palette (light mode, print):
  slot 1 blue  #2a78d6  unedited model
  slot 2 orange #eb6834 editor 3's edited model (validated as an adjacent pair: CVD dE 24.7, normal 33.6)
  muted gray   #898781  random-change control, chance / fair-share reference lines
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
# The full copy gpu_results/ (owner's Mac, not in git) if it exists, else the curated copy results/
# (fresh clone). Both hold the same experiments/analysis/ files.
DATA = ROOT / "gpu_results" / "experiments" / "analysis"
if not (ROOT / "gpu_results").exists():
    DATA = ROOT / "results" / "experiments" / "analysis"
POS = ["A", "B", "C", "D"]

# ---------------------------------------------------------------- palette and chrome (dataviz reference)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"  # categorical slot 1: unedited model
ORANGE = "#eb6834"  # categorical slot 2: editor 3's edit

CM = 1 / 2.54
WIDTH = 16 * CM

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7,
        "legend.fontsize": 7.5,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "xtick.color": INK2,
        "ytick.color": MUTED,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "svg.fonttype": "path",
        "axes.unicode_minus": True,
    }
)

# ---------------------------------------------------------------- loading


def load_json(p: Path):
    with open(p) as f:
        return json.load(f)


def load_all():
    rot = load_json(DATA / "rotation_proof.json")
    rows = [json.loads(l) for l in open(DATA / "heldout_position.jsonl") if l.strip()]
    oe = {c: load_json(DATA / "other_exams" / f"{c}.json")["results"] for c in ("unedited", "editor3", "random")}
    oe_ch = {c: load_json(DATA / "other_exams" / f"{c}_with_choices.json")["results"] for c in ("unedited", "editor3")}

    def held(cond, s):
        return [r for r in rows if r["condition"] == cond and r["set"] == s]

    # Per test: {condition: {qid: {pos: bool}}}. Random control only where it was run.
    tests = {}
    tests["hp_dev"] = {
        "unedited": {q: v["ok"] for q, v in rot["unedited_1"].items()},
        "editor3": {q: v["ok"] for q, v in rot["editor3"].items()},
        "random": {q: v["ok"] for q, v in rot["random"].items()},
    }
    for s in ("hidden", "test"):
        tests[f"hp_{s}"] = {
            "unedited": {r["id"]: r["correct_by_position"] for r in held("unedited", s)},
            "editor3": {r["id"]: r["correct_by_position"] for r in held("editor3_edit", s)},
        }
    for s in ("belebele_sv", "belebele_en", "mmlu_math"):
        tests[s] = {c: oe[c][s] for c in ("unedited", "editor3", "random")}
    return tests, rows, oe, oe_ch


TEST_ORDER = ["hp_dev", "hp_hidden", "hp_test", "belebele_sv", "belebele_en", "mmlu_math"]
TEST_LABEL = {
    "hp_dev": "Högskoleprovet dev",
    "hp_hidden": "Högskoleprovet hidden",
    "hp_test": "Högskoleprovet test 2025–26",
    "belebele_sv": "Belebele Swedish",
    "belebele_en": "Belebele English",
    "mmlu_math": "MMLU maths",
}

# ---------------------------------------------------------------- statistics


def acc_by_pos(res):
    n = len(res)
    return {p: 100 * sum(bool(v[p]) for v in res.values()) / n for p in POS}


def gained_lost(before, after):
    out = {}
    for p in POS:
        g = sum(1 for q in before if after[q][p] and not before[q][p])
        l = sum(1 for q in before if before[q][p] and not after[q][p])
        out[p] = (g, l)
    return out


def all4(res):
    return sum(all(v[p] for p in POS) for v in res.values())


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def letter_picks(chosen):
    counts = {p: 0 for p in POS}
    tot = 0
    for v in chosen.values():
        for letter in v.values():
            counts[letter] += 1
            tot += 1
    return {p: 100 * counts[p] / tot for p in POS}, tot


def heldout_methods(rows, s):
    u = [r for r in rows if r["condition"] == "unedited" and r["set"] == s]
    e = [r for r in rows if r["condition"] == "editor3_edit" and r["set"] == s]
    assert len(u) == len(e)
    return len(u), [
        ("Unedited Gemma", "unedited", sum(r["original_order_correct"] for r in u)),
        ("Prompt note", "unedited", sum(r["prompt_note_correct"] for r in u)),
        ("Option averaging", "unedited", sum(r["option_average_correct"] for r in u)),
        ("Edited Gemma", "edited", sum(r["original_order_correct"] for r in e)),
        ("Edited + averaging", "edited", sum(r["option_average_correct"] for r in e)),
    ]


# ---------------------------------------------------------------- check against CLAIMS.md

CLAIMED = {
    # G7.1 dev rotation, by position (unedited, editor 3), all-4
    ("hp_dev", "unedited"): [53.6, 75.9, 72.0, 70.8],
    ("hp_dev", "editor3"): [62.8, 73.4, 70.1, 69.9],
    # G8.1 other exams
    ("belebele_sv", "unedited"): [76.4, 82.4, 83.8, 84.6],
    ("belebele_en", "unedited"): [83.4, 88.4, 86.4, 87.2],
    ("mmlu_math", "unedited"): [45.8, 46.0, 34.5, 34.5],
    ("belebele_sv", "editor3"): [80.6, 81.2, 83.8, 83.8],
    ("belebele_en", "editor3"): [85.8, 87.8, 85.8, 86.0],
    ("mmlu_math", "editor3"): [57.0, 40.0, 32.5, 30.2],
}
CLAIMED_ALL4 = {
    ("hp_dev", "unedited"): 405, ("hp_dev", "editor3"): 439,
    ("belebele_sv", "unedited"): 345, ("belebele_en", "unedited"): 394, ("mmlu_math", "unedited"): 46,
    ("belebele_sv", "editor3"): 358, ("belebele_en", "editor3"): 392, ("mmlu_math", "editor3"): 47,
}
CLAIMED_GL = {  # G8.2, G7.1 (A only for dev, plus SUMMARY note 3 for B, C), G7.2 (A only held out)
    "belebele_sv": [(21, 0), (0, 6), (2, 2), (1, 5)],
    "belebele_en": [(12, 0), (1, 4), (0, 3), (2, 8)],
    "mmlu_math": [(45, 0), (2, 26), (2, 10), (0, 17)],
    "hp_dev": [(83, 0), (4, 26), (5, 22), None],
    "hp_hidden": [(27, 0), None, None, None],
    "hp_test": [(11, 0), None, None, None],
}
CLAIMED_HELD_A = {("hp_hidden", "unedited"): 57.2, ("hp_hidden", "editor3"): 65.6,
                  ("hp_test", "unedited"): 57.5, ("hp_test", "editor3"): 66.7}
CLAIMED_METHODS = {"hidden": [232, 230, 238, 236, 238], "test": [84, 88, 90, 85, 90]}
CLAIMED_PICKS = {
    ("belebele_sv", "unedited"): [22.2, 25.0, 25.6, 27.2], ("belebele_sv", "editor3"): [24.1, 24.1, 25.1, 26.8],
    ("belebele_en", "unedited"): [23.1, 25.9, 25.1, 26.0], ("belebele_en", "editor3"): [24.8, 25.4, 24.6, 25.2],
    ("mmlu_math", "unedited"): [28.9, 29.5, 20.2, 21.4], ("mmlu_math", "editor3"): [39.2, 24.3, 18.5, 18.0],
}


def check_claims(tests, rows, oe_ch):
    bad = []

    def cmp(label, got, want, tol=0.05):
        if abs(got - want) > tol + 1e-9:
            bad.append(f"{label}: computed {got:.2f}, CLAIMS.md {want}")

    for (t, c), want in CLAIMED.items():
        got = acc_by_pos(tests[t][c])
        for p, w in zip(POS, want):
            cmp(f"{t} {c} accuracy at {p}", got[p], w)
    for (t, c), want in CLAIMED_ALL4.items():
        cmp(f"{t} {c} right in all 4 positions", all4(tests[t][c]), want, 0)
    for t, want in CLAIMED_GL.items():
        got = gained_lost(tests[t]["unedited"], tests[t]["editor3"])
        for p, w in zip(POS, want):
            if w is not None and got[p] != w:
                bad.append(f"{t} gained/lost at {p}: computed {got[p]}, CLAIMS.md {w}")
    for (t, c), want in CLAIMED_HELD_A.items():
        cmp(f"{t} {c} accuracy at A", acc_by_pos(tests[t][c])["A"], want)
    for s, want in CLAIMED_METHODS.items():
        _, ms = heldout_methods(rows, s)
        for (name, _, k), w in zip(ms, want):
            cmp(f"held-out {s} {name} count", k, w, 0)
    for (s, c), want in CLAIMED_PICKS.items():
        got, _ = letter_picks(oe_ch[c][s + "/chosen"])
        for p, w in zip(POS, want):
            cmp(f"{s} {c} share picking {p}", got[p], w)
    # Random control: within one question per position of unedited (SUMMARY correction to G8.1g).
    # Dev random control at A: gained 2, lost 0 (G7.1).
    if gained_lost(tests["hp_dev"]["unedited"], tests["hp_dev"]["random"])["A"] != (2, 0):
        bad.append("hp_dev random control at A is not gained 2 / lost 0")
    for t in ("belebele_sv", "belebele_en", "mmlu_math"):
        u, r = acc_by_pos(tests[t]["unedited"]), acc_by_pos(tests[t]["random"])
        n = len(tests[t]["unedited"])
        for p in POS:
            if abs(u[p] - r[p]) > 100 / n + 1e-9:
                bad.append(f"{t} random control at {p} differs from unedited by {r[p]-u[p]:+.2f} points")
    # The *_with_choices files must agree with the files used for accuracy.
    for c in ("unedited", "editor3"):
        for s in ("belebele_sv", "belebele_en", "mmlu_math"):
            if oe_ch[c][s] != tests[s][c]:
                bad.append(f"{c}_with_choices.json {s} correctness differs from {c}.json")
    return bad


# ---------------------------------------------------------------- drawing helpers


def style_axes(ax, ygrid=True, xgrid=False):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", length=0, pad=3)
    ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.6, linestyle="-")


def rounded_bar(ax, x, width, y0, y1, color, radius_pt=1.6, zorder=3):
    """Column from baseline y0 to data end y1: square at the baseline, rounded data end.

    Axes limits and position must be final before calling (radius is converted from points)."""
    if y1 == y0:
        return
    fig = ax.figure
    bbox = ax.get_window_extent()
    x0l, x1l = ax.get_xlim()
    y0l, y1l = ax.get_ylim()
    px = radius_pt * fig.dpi / 72
    rx = px * (x1l - x0l) / bbox.width
    ry = px * (y1l - y0l) / bbox.height
    rx = min(rx, width / 2)
    ry = min(ry, abs(y1 - y0))
    s = 1 if y1 > y0 else -1
    L, R = x - width / 2, x + width / 2
    verts = [
        (L, y0), (L, y1 - s * ry), (L, y1), (L + rx, y1),
        (R - rx, y1), (R, y1), (R, y1 - s * ry), (R, y0), (L, y0),
    ]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=zorder))


def save(fig, name):
    for ext in ("png", "svg"):
        kw = {"dpi": 200} if ext == "png" else {}
        fig.savefig(HERE / f"{name}.{ext}", **kw)
    plt.close(fig)
    print(f"wrote {name}.png, {name}.svg")


def panel_grid(nrows, ncols, height_cm, left=0.075, right=0.985, top=0.86, bottom=0.12, wspace=0.28, hspace=0.75):
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH, height_cm * CM), squeeze=False)
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)
    return fig, axes


# ---------------------------------------------------------------- figure 1


def fig1(tests):
    fig, axes = panel_grid(2, 3, 10.2, top=0.77, bottom=0.075, hspace=0.80)
    fig.text(0.075, 0.965, "On reading tests, Unedited Gemma does worst when the right answer is option A",
             fontsize=9.5, fontweight="bold", ha="left", va="top")
    fig.text(0.075, 0.915, "Accuracy (%) by position of the correct answer. Every question asked four times "
             "with its options rotated.\nGray line: chance (25%). Maths reverses the pattern: options A and B are its easiest.", fontsize=7.5, color=INK2, ha="left", va="top")
    for ax, t in zip(axes.flat, TEST_ORDER):
        res = tests[t]["unedited"]
        acc = acc_by_pos(res)
        style_axes(ax)
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(range(4), POS)
        ax.set_title(f"{TEST_LABEL[t]}\n", loc="left", pad=2)
        ax.text(0, 1.035, f"n = {len(res)} questions", transform=ax.transAxes, fontsize=7, color=INK2,
                ha="left", va="bottom")
        ax.axhline(25, color=MUTED, linewidth=0.6, zorder=2)
    fig.canvas.draw()
    for ax, t in zip(axes.flat, TEST_ORDER):
        acc = acc_by_pos(tests[t]["unedited"])
        for i, p in enumerate(POS):
            rounded_bar(ax, i, 0.56, 0, acc[p], BLUE)
            ax.text(i, acc[p] + 2, f"{acc[p]:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (%)")
    save(fig, "fig1_position_by_test")


# ---------------------------------------------------------------- figure 2


def fig2(tests):
    fig, axes = panel_grid(2, 3, 11.5, top=0.75, bottom=0.10, hspace=0.95)
    fig.text(0.075, 0.975, "Edited Gemma does better at option A on every test, at a small cost at options B–D",
             fontsize=9.5, fontweight="bold", ha="left", va="top")
    fig.text(0.075, 0.93, "Change in accuracy (percentage points), Edited Gemma minus Unedited Gemma, by position of the correct answer.\n"
             "Edited Gemma = Gemma with the AI editor's weight edit. Under each letter: questions that became right / became wrong.", fontsize=7.5, color=INK2,
             ha="left", va="top")
    handles = [Patch(facecolor=ORANGE, label="Edited Gemma"),
               Line2D([], [], marker="o", linestyle="none", markersize=4.5, markerfacecolor=MUTED,
                      markeredgecolor=SURFACE, markeredgewidth=1.2, label="Random change of the same size")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.068, 0.875), ncol=2, frameon=False,
               handlelength=1.1, columnspacing=1.8)
    YMIN, YMAX = -9, 15
    for ax, t in zip(axes.flat, TEST_ORDER):
        u, e = tests[t]["unedited"], tests[t]["editor3"]
        gl = gained_lost(u, e)
        style_axes(ax)
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(YMIN, YMAX)
        ax.set_yticks([-5, 0, 5, 10, 15])
        ax.set_yticklabels(["−5", "0", "+5", "+10", "+15"])
        ax.set_xticks(range(4), [f"{p}\n+{gl[p][0]}/−{gl[p][1]}" for p in POS])
        ax.tick_params(axis="x", labelsize=6.8)
        ax.spines["bottom"].set_visible(False)
        ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=2)
        ax.set_title(f"{TEST_LABEL[t]}\n", loc="left", pad=2)
        note = "" if "random" in tests[t] else " · no random control"
        ax.text(0, 1.035, f"n = {len(u)} questions{note}", transform=ax.transAxes, fontsize=7, color=INK2,
                ha="left", va="bottom")
    fig.canvas.draw()
    for ax, t in zip(axes.flat, TEST_ORDER):
        au, ae = acc_by_pos(tests[t]["unedited"]), acc_by_pos(tests[t]["editor3"])
        for i, p in enumerate(POS):
            d = ae[p] - au[p]
            rounded_bar(ax, i, 0.5, 0, d, ORANGE)
            va, off = ("bottom", 0.6) if d >= 0 else ("top", -0.6)
            label = "0" if abs(d) < 0.05 else f"{d:+.1f}".replace("-", "−")
            ax.text(i, d + off, label, ha="center", va=va, fontsize=6.5, color=INK2)
            if "random" in tests[t]:
                dr = acc_by_pos(tests[t]["random"])[p] - au[p]
                ax.plot([i + 0.38], [dr], marker="o", markersize=4.5, markerfacecolor=MUTED,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4, clip_on=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Change (points)")
    save(fig, "fig2_edit_effect")


# ---------------------------------------------------------------- figure 3


def fig3(rows):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 8.0 * CM), sharey=True)
    fig.subplots_adjust(left=0.175, right=0.985, top=0.66, bottom=0.14, wspace=0.10)
    fig.text(0.015, 0.975, "On held-out questions, all five ways of answering score within noise of each other",
             fontsize=9.5, fontweight="bold", ha="left", va="top")
    fig.text(0.015, 0.905, "Accuracy (%) on Högskoleprovet LÄS + MEK questions, options in original order unless averaged."
             "\nHorizontal bars: 95% Wilson intervals. Vertical line: Unedited Gemma. Edited Gemma = Gemma with the AI editor's weight edit.", fontsize=7.5, color=INK2,
             ha="left", va="top")
    handles = [Line2D([], [], marker="o", color=BLUE, linewidth=1.5, markersize=5, label="Unedited Gemma"),
               Line2D([], [], marker="o", color=ORANGE, linewidth=1.5, markersize=5, label="Edited Gemma")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 0.81), ncol=2, frameon=False,
               handlelength=1.6, columnspacing=1.8)
    titles = {"hidden": "Hidden set", "test": "Test set (2025–26 papers)"}
    for ax, s in zip(axes, ("hidden", "test")):
        n, ms = heldout_methods(rows, s)
        style_axes(ax, ygrid=False, xgrid=True)
        ax.spines["bottom"].set_visible(False)
        ax.set_xlim(55, 93)
        ax.set_xticks([60, 70, 80, 90])
        ax.set_ylim(len(ms) - 0.4, -0.7)
        ax.set_yticks(range(len(ms)), [m[0] for m in ms])
        ax.tick_params(axis="y", labelsize=7.5, labelcolor=INK)
        base = 100 * ms[0][2] / n
        ax.axvline(base, color=BASELINE, linewidth=0.8, zorder=1)
        for i, (name, model, k) in enumerate(ms):
            col = BLUE if model == "unedited" else ORANGE
            lo, hi = wilson(k, n)
            p = 100 * k / n
            ax.plot([lo, hi], [i, i], color=col, linewidth=1.5, solid_capstyle="round", zorder=3)
            ax.plot([p], [i], marker="o", markersize=5.5, markerfacecolor=col, markeredgecolor=SURFACE,
                    markeredgewidth=1.4, zorder=4)
            ax.text(hi + 0.7, i, f"{p:.1f}  ({k}/{n})", va="center", ha="left", fontsize=6.5, color=INK2)
        ax.set_title(f"{titles[s]}, n = {n}", loc="left", pad=6)
        ax.set_xlabel("Accuracy (%)")
    save(fig, "fig3_methods_heldout")


# ---------------------------------------------------------------- figure 4


def fig4(oe_ch):
    sets = ["belebele_sv", "belebele_en", "mmlu_math"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 7.8 * CM), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.66, bottom=0.09, wspace=0.12)
    fig.text(0.075, 0.975, "Edited Gemma picks option A more often, and on maths far more than a fair share",
             fontsize=9.5, fontweight="bold", ha="left", va="top")
    fig.text(0.075, 0.905, "Share of all answers (%) that chose each option (A to D), over the four option rotations.\n"
             "Edited Gemma = Gemma with the AI editor's weight edit.",
             fontsize=7.5, color=INK2, ha="left", va="top")
    handles = [Patch(facecolor=BLUE, label="Unedited Gemma"), Patch(facecolor=ORANGE, label="Edited Gemma"),
               Line2D([], [], color=MUTED, linewidth=0.8, label="25% (fair share)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.068, 0.835), ncol=3, frameon=False,
               handlelength=1.1, columnspacing=1.8)
    W = 0.34
    for ax, s in zip(axes, sets):
        style_axes(ax)
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(0, 45)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_xticks(range(4), POS)
        ax.axhline(25, color=MUTED, linewidth=0.8, zorder=4)
        nq = len(oe_ch["unedited"][s + "/chosen"])
        ax.set_title(f"{TEST_LABEL[s]}\n", loc="left", pad=2)
        ax.text(0, 1.035, f"n = {nq} questions × 4 = {4 * nq} answers", transform=ax.transAxes, fontsize=7,
                color=INK2, ha="left", va="bottom")
    axes[0].set_ylabel("Share of answers (%)")
    fig.canvas.draw()
    for ax, s in zip(axes, sets):
        pu, _ = letter_picks(oe_ch["unedited"][s + "/chosen"])
        pe, _ = letter_picks(oe_ch["editor3"][s + "/chosen"])
        for i, p in enumerate(POS):
            # 0.04 data units of surface between the pair = the 2px gap at this size
            rounded_bar(ax, i - W / 2 - 0.02, W, 0, pu[p], BLUE)
            rounded_bar(ax, i + W / 2 + 0.02, W, 0, pe[p], ORANGE)
        # Label the A pair only, as one 'before -> after' label (the story); the axis carries the rest, captions.md gives all values.
        ytxt = max(pu["A"], pe["A"]) + 1.2
        if 22.5 < ytxt < 27.5:  # keep clear of the 25% line
            ytxt = 27.5
        ax.text(0.05, ytxt, f"{pu['A']:.1f} $\\rightarrow$ {pe['A']:.1f}", ha="center", va="bottom", fontsize=6.5,
                color=INK2)
    save(fig, "fig4_letter_picks")


# ---------------------------------------------------------------- main



# ---------------------------------------------------------------- figure 5


def fig5(tests):
    """Unedited vs edited Gemma (editor 3's weight edit), accuracy by answer position, on every test."""
    fig, axes = panel_grid(2, 3, 11.6, top=0.72, bottom=0.07, hspace=0.95)
    fig.text(0.075, 0.975, "Edited Gemma does better at option A on every test, but its average barely moves",
             fontsize=9.5, fontweight="bold", ha="left", va="top")
    fig.text(0.075, 0.925, "Accuracy (%) by position of the correct answer. Every question asked four times with its options rotated.\n"
             "Edited Gemma = Gemma with the AI editor's weight edit. Under each title: questions, and average over options A to D.", fontsize=7.5, color=INK2, ha="left", va="top")
    handles = [Patch(facecolor=BLUE, label="Unedited Gemma"),
               Patch(facecolor=ORANGE, label="Edited Gemma")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.068, 0.885), ncol=2, frameon=False,
               fontsize=7.5, handlelength=1.2, columnspacing=1.6)
    for ax, t in zip(axes.flat, TEST_ORDER):
        u, e = tests[t]["unedited"], tests[t]["editor3"]
        au, ae = acc_by_pos(u), acc_by_pos(e)
        mu, me = sum(au.values()) / 4, sum(ae.values()) / 4
        style_axes(ax)
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(range(4), POS)
        ax.set_title(f"{TEST_LABEL[t]}\n", loc="left", pad=2)
        ax.text(0, 1.035, f"{len(u)} questions \u00b7 avg {round(mu + 1e-9, 1):.1f} to {round(me + 1e-9, 1):.1f}",
                transform=ax.transAxes, fontsize=7, color=INK2, ha="left", va="bottom")
    fig.canvas.draw()
    w = 0.36
    for ax, t in zip(axes.flat, TEST_ORDER):
        au, ae = acc_by_pos(tests[t]["unedited"]), acc_by_pos(tests[t]["editor3"])
        for i, p in enumerate(POS):
            rounded_bar(ax, i - w / 2 - 0.01, w, 0, au[p], BLUE)
            rounded_bar(ax, i + w / 2 + 0.01, w, 0, ae[p], ORANGE)
            ax.text(i - w / 2 - 0.01, au[p] + 2, f"{au[p]:.0f}", ha="center", va="bottom", fontsize=5.5, color=INK2)
            ax.text(i + w / 2 + 0.01, ae[p] + 2, f"{ae[p]:.0f}", ha="center", va="bottom", fontsize=5.5, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (%)")
    save(fig, "fig5_unedited_vs_edited")



# ---------------------------------------------------------------- figure 6


def fig6(tests):
    """Högskoleprovet dev only: right answers out of 900 by answer position, before and after the weight edit."""
    u, e = tests["hp_dev"]["unedited"], tests["hp_dev"]["editor3"]
    n = len(u)
    cu = {p: sum(bool(v[p]) for v in u.values()) for p in POS}
    ce = {p: sum(bool(v[p]) for v in e.values()) for p in POS}
    gl = gained_lost(u, e)
    fig, axes = panel_grid(1, 1, 9.2, left=0.09, right=0.98, top=0.66, bottom=0.16)
    ax = axes[0, 0]
    fig.text(0.09, 0.965, "Same 900 questions, with the right answer rotated between options A, B, C and D",
             fontsize=9.5, fontweight="bold", ha="left", va="top")
    fig.text(0.09, 0.895, "Questions answered right (out of 900) with the right answer placed at option A, B, C or D.\n"
             "Högskoleprovet reading and sentence completion, older papers. Edited Gemma = Gemma with the AI editor's\n"
             "weight edit. Under each letter: questions it turned right / turned wrong.", fontsize=7.5, color=INK2, ha="left", va="top")
    handles = [Patch(facecolor=BLUE, label="Unedited Gemma"),
               Patch(facecolor=ORANGE, label="Edited Gemma")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.083, 0.765), ncol=2, frameon=False,
               fontsize=7.5, handlelength=1.2, columnspacing=1.6)
    style_axes(ax)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, n)
    ax.set_yticks([0, 225, 450, 675, 900])
    ax.set_xticks(range(4), [f"Option {p}\n+{gl[p][0]} / \u2212{gl[p][1]}" for p in POS])
    ax.set_ylabel("Right answers (of 900)")
    fig.canvas.draw()
    w = 0.36
    for i, p in enumerate(POS):
        rounded_bar(ax, i - w / 2 - 0.01, w, 0, cu[p], BLUE)
        rounded_bar(ax, i + w / 2 + 0.01, w, 0, ce[p], ORANGE)
        ax.text(i - w / 2 - 0.01, cu[p] + 12, f"{cu[p]}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.text(i + w / 2 + 0.01, ce[p] + 12, f"{ce[p]}", ha="center", va="bottom", fontsize=7, color=INK2)
    save(fig, "fig6_position_before_after")



# ---------------------------------------------------------------- figure 7


def choice_category(r):
    """Same rules as analysis/choice_patterns.py."""
    o, ans = r["orders"], r["answer"]
    right = {L: o[L]["chosen"] == ans for L in POS}
    k = sum(right.values())
    if k == 4:
        return "knows"
    if not right["A"] and all(right[L] for L in "BCD"):
        return "habit_A"
    if k > 0:
        return "unsure"
    if len({o[L]["chosen"] for L in POS}) == 1:
        return "wrong_belief"
    if len({o[L]["chosen_position"] for L in POS}) == 1:
        return "position_guess"
    return "no_idea"


def fig7():
    """What Gemma picks across the four orders: five kinds of question, before and after the weight edit."""
    R = load_json(DATA / "rotation_choices.json")
    ids = [i for i, r in R["unedited"].items() if r["split"] == "dev"]
    cu = {c: 0 for c in ("knows", "habit_A", "unsure", "wrong_belief", "no_idea", "position_guess")}
    ce = dict(cu)
    for i in ids:
        cu[choice_category(R["unedited"][i])] += 1
        ce[choice_category(R["editor3"][i])] += 1
    assert cu["position_guess"] == 0 and ce["position_guess"] == 0
    GREEN, RED, PALE = "#2e8b57", "#d64545", "#f6dede"
    rows = [  # key, title, explanation, pattern (True = right), marks inside the squares
        ("knows", "Knows it", "Right answer in all four tries.", [1, 1, 1, 1], ["", "", "", ""]),
        ("habit_A", "Knows it, but avoids option A", "Right at options B, C and D. Wrong only\nwhen the right answer is option A.",
         [0, 1, 1, 1], ["x", "", "", ""]),
        ("unsure", "Unsure", "Right in some tries, wrong in others.\nThe order of the options tips it.",
         [1, 0, 1, 0], ["", "x", "", "y"]),
        ("wrong_belief", "Believes something wrong", "Wrong every time, and picks the SAME\nwrong option every time (99% sure).",
         [0, 0, 0, 0], ["x", "x", "x", "x"]),
        ("no_idea", "Doesn't know", "Wrong every time, and which wrong\noption it picks changes with the order.",
         [0, 0, 0, 0], ["x", "y", "z", "x"]),
    ]
    fig = plt.figure(figsize=(WIDTH, 11.2 * CM))
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.80])
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.4, len(rows) + 0.35)
    ax.axis("off")
    fig.text(0.04, 0.975, "Each question asked four times, with the right answer moved between options A, B, C and D",
             fontsize=9.5, fontweight="bold", ha="left", va="top")
    fig.text(0.04, 0.925, "What Gemma picks across the four tries sorts the 900 older-paper questions into five kinds.\n"
             "Edited Gemma has the AI editor's weight edit. Green: right option. Red: wrong option (x, y, z show which).",
             fontsize=7.5, color=INK2, ha="left", va="top")
    top = len(rows) - 0.25
    for k, L in enumerate(POS):
        ax.text(0.85 + k * 0.62, top + 0.42, L, ha="center", va="center", fontsize=7.5, color=INK2, fontweight="bold")
    ax.text(0.65, top + 0.75, "Right answer at option", ha="left", va="center", fontsize=7, color=INK2)
    ax.text(10.15, top + 0.75, "Questions (of 900)", ha="left", va="center", fontsize=7, color=INK2)
    handles = [Patch(facecolor=BLUE, label="Unedited Gemma"), Patch(facecolor=ORANGE, label="Edited Gemma")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.655, 1.035), ncol=2, frameon=False,
              fontsize=7, handlelength=1.0, columnspacing=1.0)
    scale = 5.2 / 450
    for n, (key, title, expl, pat, marks) in enumerate(rows):
        y = top - n
        for k in range(4):
            x = 0.85 + k * 0.62
            ok = pat[k]
            ax.add_patch(plt.Rectangle((x - 0.25, y - 0.25), 0.5, 0.5, facecolor=GREEN if ok else PALE,
                                       edgecolor=GREEN if ok else RED, linewidth=0.8))
            if marks[k]:
                ax.text(x, y, marks[k], ha="center", va="center", fontsize=7.5, color=RED, fontweight="bold")
        ax.text(3.5, y + 0.14, title, ha="left", va="center", fontsize=8, fontweight="bold", color=INK)
        ax.text(3.5, y - 0.04, expl, ha="left", va="top", fontsize=6.6, color=INK2, linespacing=1.15)
        for off, val, col in ((0.13, cu[key], BLUE), (-0.13, ce[key], ORANGE)):
            ax.add_patch(plt.Rectangle((10.15, y + off - 0.1), val * scale, 0.2, facecolor=col, edgecolor="none"))
            ax.text(10.15 + val * scale + 0.1, y + off, f"{val}", ha="left", va="center", fontsize=6.8, color=INK2)
    # Highlight the one kind the weight edit clearly changed.
    yh = top - [r[0] for r in rows].index("habit_A")
    ax.add_patch(FancyBboxPatch((9.95, yh - 0.36), 2.55, 0.72, boxstyle="round,pad=0.02,rounding_size=0.18",
                                facecolor="#fff3c4", edgecolor="#d9a400", linewidth=1.3, zorder=0))
    ax.text(12.7, yh, "The weight edit\nhelps here:\n124 to 76", ha="left", va="center", fontsize=7,
            color="#8a6a00", fontweight="bold", linespacing=1.15)
    ax.text(0.65, -0.3, "No question was answered by picking the same letter every time regardless of the text "
            "(0 before, 0 after).", ha="left", va="center", fontsize=6.8, color=INK2, style="italic")
    save(fig, "fig7_question_kinds")



# ---------------------------------------------------------------- figure 8


def fig8():
    """Compression from 16-bit to 4-bit: right answers at option A on the 440 unseen questions (hidden + test papers)."""
    q4 = {}
    for f in sorted((DATA / "q4_rotation").glob("*_shard*.jsonl")):
        cond = f.name.split("_shard")[0]
        for line in open(f):
            r = json.loads(line)
            q4.setdefault(cond, {})[(r["set"], r["id"])] = r["correct_by_position"]
    full = {}
    for line in open(DATA / "heldout_position.jsonl"):
        r = json.loads(line)
        if r["condition"] in ("unedited", "editor3_edit"):
            full.setdefault(r["condition"], {})[(r["set"], r["id"])] = r["correct_by_position"]

    def right_a(d):
        return sum(bool(v["A"]) for v in d.values())

    def gained_lost_a(a, b):
        return sum(b[k]["A"] and not a[k]["A"] for k in a), sum(a[k]["A"] and not b[k]["A"] for k in a)

    n = len(full["unedited"])
    assert n == len(q4["unedited"]) == len(q4["edited"]) == len(q4["control"])
    rows = [
        ("16-bit (full size)", right_a(full["unedited"]), right_a(full["editor3_edit"]), ORANGE,
         gained_lost_a(full["unedited"], full["editor3_edit"])),
        ("4-bit (compressed)", right_a(q4["unedited"]), right_a(q4["edited"]), ORANGE,
         gained_lost_a(q4["unedited"], q4["edited"])),
        ("4-bit, random change\nof the same size", right_a(q4["unedited"]), right_a(q4["control"]), MUTED,
         gained_lost_a(q4["unedited"], q4["control"])),
    ]
    fig = plt.figure(figsize=(WIDTH, 8.8 * CM))
    fig.text(0.04, 0.965, "The weight edit still works after compression to 4-bit",
             fontsize=9.5, fontweight="bold", ha="left", va="top")
    fig.text(0.04, 0.89, f"Questions answered right when the right answer is option A, out of {n} questions no edit was "
             "built on\n(8 hidden older papers and 3 test papers). Solid arrows: Unedited Gemma to the changed model. "
             "Dashed arrow: compression.\nFixed: wrong before, right after. Broken: right before, wrong after.",
             fontsize=7.5, color=INK2, ha="left", va="top")
    ax = fig.add_axes([0.25, 0.25, 0.56, 0.48])
    style_axes(ax, ygrid=False, xgrid=True)
    ax.spines["bottom"].set_visible(False)
    lo, hi = 178, 300
    ax.set_xlim(lo, hi)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ticks = [200, 220, 240, 260, 280, 300]
    ax.set_xticks(ticks, [f"{t}\n({100 * t / n:.0f}%)" for t in ticks])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", labelsize=8, labelcolor=INK)
    ax.set_xlabel(f"Right at option A (out of {n})", fontsize=7.5, color=INK2, labelpad=4)
    for y, (label, a, b, col, (g, l)) in enumerate(rows):
        if b != a:
            ax.annotate("", xy=(b, y), xytext=(a, y),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, shrinkA=5, shrinkB=5, mutation_scale=10))
        ax.plot([a], [y], "o", color=BLUE, markersize=7, zorder=4)
        ax.plot([b], [y], "o", color=col, markersize=7, zorder=5)
        ax.text(a - 2, y - 0.28, f"{a}", ha="right", va="center", fontsize=7, color=BLUE)
        ax.text(b + 2, y - 0.28, f"{b}", ha="left", va="center", fontsize=7.5 if col != MUTED else 7,
                color=col if col != MUTED else MUTED, fontweight="bold" if col != MUTED else "normal", zorder=6)
        verdict = f"{g} questions fixed,\n{l} broken"
        edited = col != MUTED
        if edited:  # highlight what matters: where Edited Gemma lands
            ax.add_patch(FancyBboxPatch((b - 7, y - 0.50), 17, 0.82, boxstyle="round,pad=0.0,rounding_size=2.5",
                                        facecolor="#fff3c4", edgecolor="#d9a400", linewidth=1.2, zorder=1,
                                        mutation_aspect=0.12))
        fig.text(0.83, ax.transData.transform((0, y))[1] / fig.dpi / (8.8 * CM), verdict,
                 ha="left", va="center", fontsize=7.5, color=col if edited else MUTED, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff3c4", edgecolor="#d9a400", linewidth=1.0)
                 if edited else None)
    u16, u4 = rows[0][1], rows[1][1]
    # Compression itself: Unedited Gemma 16-bit -> 4-bit (fewer picks of option A when it is the right answer).
    ax.annotate("", xy=(u4, 1), xytext=(u16, 0),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.3, linestyle=(0, (3, 2)), shrinkA=6, shrinkB=6,
                                mutation_scale=9, connectionstyle="arc3,rad=0.3"))
    ax.text(lo + 2, 0.4, f"Compression:\n{u16 - u4} fewer picks\nof option A", ha="left", va="center",
            fontsize=6.8, color=BLUE, fontweight="bold", linespacing=1.1)
    e4 = rows[1][2]
    fig.text(0.04, 0.035, f"Edited Gemma at 4-bit ({e4}) gets more questions right at option A than Unedited Gemma "
             f"at 16-bit ({u16}).", ha="left", va="bottom", fontsize=7.5, color="#8a6a00", fontweight="bold")
    handles = [Line2D([], [], marker="o", linestyle="none", color=BLUE, markersize=6, label="Unedited Gemma"),
               Line2D([], [], marker="o", linestyle="none", color=ORANGE, markersize=6, label="Edited Gemma"),
               Line2D([], [], marker="o", linestyle="none", color=MUTED, markersize=6, label="Random change")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.035, 0.785), ncol=3, frameon=False, fontsize=7.5,
               handletextpad=0.3, columnspacing=1.6)
    save(fig, "fig8_compression")


# ---------------------------------------------------------------- figure 9


def fig9():
    """Fair comparison: share of tries in which Gemma picks option A when it is right vs wrong, and how often a
    pick of option A is right. Older papers (dev), every question asked in all four orders."""
    R = load_json(DATA / "rotation_choices.json")
    ids = [i for i, r in R["unedited"].items() if r["split"] == "dev"]
    n = len(ids)

    def count(cond):
        right = sum(R[cond][i]["orders"]["A"]["chosen_position"] == "A" for i in ids)
        wrong = sum(R[cond][i]["orders"][p]["chosen_position"] == "A" for i in ids for p in "BCD")
        return right, wrong

    (ru, wu), (re_, we) = count("unedited"), count("editor3")
    panels = [
        ("When option A is right", f"share of the {n} tries that pick option A",
         100 * ru / n, 100 * re_ / n, f"{ru} of {n}", f"{re_} of {n}"),
        ("When option A is wrong", f"share of the {3 * n:,} tries that pick option A",
         100 * wu / (3 * n), 100 * we / (3 * n), f"{wu} of {3 * n:,}", f"{we} of {3 * n:,}"),
        ("When Gemma picks option A", "share of those picks that are right",
         100 * ru / (ru + wu), 100 * re_ / (re_ + we), f"{ru} of {ru + wu}", f"{re_} of {re_ + we}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 8.6 * CM), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.58, bottom=0.13, wspace=0.18)
    fig.text(0.04, 0.965, "The weight edit makes Gemma pick option A more often, not judge it better",
             fontsize=9.5, fontweight="bold", ha="left", va="top")
    fig.text(0.04, 0.895, f"{n} older-paper questions, each asked four times with the options shuffled. Same scale in every panel.\n"
             "Left and middle: Edited Gemma picks option A more often, mostly where option A is the right answer.\n"
             "Right: but when Edited Gemma picks option A, it is right less often than Unedited Gemma.",
             fontsize=7.5, color=INK2, ha="left", va="top")
    handles = [Patch(facecolor=BLUE, label="Unedited Gemma"), Patch(facecolor=ORANGE, label="Edited Gemma")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.035, 0.735), ncol=2, frameon=False, fontsize=7.5,
               handlelength=1.2, columnspacing=1.6)
    for k, (ax, (title, sub, u, e, lu, le)) in enumerate(zip(axes, panels)):
        style_axes(ax)
        ax.set_xlim(-0.7, 1.7)
        ax.set_ylim(0, 114)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks([0, 1], ["Unedited\nGemma", "Edited\nGemma"])
        ax.set_title(f"{title}\n", loc="left", pad=2, fontsize=8)
        ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=6.5, color=INK2, ha="left", va="bottom")
        if k == 0:
            ax.set_ylabel("Percent")
        fig.canvas.draw()
        for x, v, col, lab in ((0, u, BLUE, lu), (1, e, ORANGE, le)):
            rounded_bar(ax, x, 0.55, 0, v, col)
            ax.text(x, v + 2, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5, color=INK, fontweight="bold")
            ax.text(x, v + 9, lab, ha="center", va="bottom", fontsize=5.8, color=INK2)
        d = e - u
        ax.text(0.5, 107, f"{d:+.1f} points".replace("-", "\u2212"), ha="center", va="center", fontsize=7.5, fontweight="bold",
                color="#8a6a00", bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3c4", edgecolor="#d9a400",
                                           linewidth=1.0))
    save(fig, "fig9_option_a_picks")


# ---------------------------------------------------------------- figure 10


def fig10():
    """Schematic (not measured): how the AI editor's weight edit changes where layer 23 looks."""
    GOLD, GOLDEDGE, GOLDINK = "#fff3c4", "#d9a400", "#8a6a00"
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 7.4 * CM))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.74, bottom=0.10, wspace=0.08)
    fig.text(0.03, 0.965, "How the AI editor's weight edit works (schematic)", fontsize=9.5, fontweight="bold",
             ha="left", va="top")
    fig.text(0.03, 0.895, "In layer 23, two attention heads look back at the four options and copy the letter of the "
             "one they look at most\ninto the answer. Line width shows how much they look at each option. "
             "Illustration only, not measured values.", fontsize=7.5, color=INK2, ha="left", va="top")
    widths = {"before": [0.8, 3.2, 3.0, 2.8], "after": [3.2, 3.0, 2.8, 2.7]}
    for ax, key, title in ((axes[0], "before", "Before the weight edit"), (axes[1], "after", "After the weight edit")):
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.0, 4.6)
        ax.axis("off")
        ax.text(0.2, 4.35, title, fontsize=8.5, fontweight="bold", color=INK, ha="left", va="center")
        ys = [3.4, 2.4, 1.4, 0.4]
        for k, (y, L) in enumerate(zip(ys, POS)):
            hl = key == "after" and L == "A"
            ax.add_patch(FancyBboxPatch((0.3, y - 0.32), 2.9, 0.64, boxstyle="round,pad=0.02,rounding_size=0.12",
                                        facecolor=GOLD if hl else "#f1f0ec", edgecolor=GOLDEDGE if hl else BASELINE,
                                        linewidth=1.2 if hl else 0.8))
            ax.text(0.55, y, f"Option {L}", fontsize=7.5, color=INK, ha="left", va="center",
                    fontweight="bold" if hl else "normal")
            col = ORANGE if key == "after" else BLUE
            ax.plot([3.25, 7.0], [y, 1.9], color=col, linewidth=widths[key][k], alpha=0.85, solid_capstyle="round")
        ax.add_patch(plt.Circle((7.75, 1.9), 0.72, facecolor="white", edgecolor=INK2, linewidth=1.0))
        ax.text(7.75, 2.05, "Answer", fontsize=7.5, ha="center", va="center", color=INK, fontweight="bold")
        ax.text(7.75, 1.7, "layer 23", fontsize=6.3, ha="center", va="center", color=INK2)
        if key == "before":
            ax.text(5.1, -0.55, "Option A gets little attention, so Gemma rarely\npicks it, even when it is right.",
                    fontsize=6.8, color=INK2, ha="center", va="center", linespacing=1.15)
        else:
            ax.text(5.1, -0.55, "The edit makes option A easier to match, so the heads\nlook at it more. "
                    "Rank-1 change to 256 rows of the key matrix.", fontsize=6.8, color=GOLDINK, ha="center",
                    va="center", fontweight="bold", linespacing=1.15)
    save(fig, "fig10_how_edit_works")


def main():
    tests, rows, oe, oe_ch = load_all()
    bad = check_claims(tests, rows, oe_ch)
    if bad:
        print("MISMATCHES with verification/CLAIMS.md:")
        for b in bad:
            print("  -", b)
    else:
        print("All recomputed values match verification/CLAIMS.md.")
    fig1(tests)
    fig2(tests)
    fig3(rows)
    fig4(oe_ch)
    fig5(tests)
    fig6(tests)
    fig7()
    fig8()
    fig9()
    fig10()


if __name__ == "__main__":
    main()
