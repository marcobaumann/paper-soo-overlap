"""
make_waterfall.py — Board slide: decomposition of the deception-reduction effect.

Waterfall showing that most of the honesty effect in our Mistral-7B reproduction
comes from scenario EXPOSURE, not the SOO objective:
  Generic FT (02) 93.76%  --exposure--> 17.60% (03)  --SOO--> 1.60% (01)

All numbers are our own, dose-matched, 5 seeds, Sonnet-judged. Regenerate with:
    python make_waterfall.py
Outputs control_ladder_waterfall.png next to this script.

Palette: validated slots from the dataviz reference palette (blue #2a78d6,
orange #eb6834; ALL CHECKS PASS, light surface). Single measure (deceptive
rate); the two decrements are categorical (exposure vs SOO), each also directly
labeled so identity is never color-alone.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- palette (validated) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
BLUE = "#2a78d6"      # SOO contribution / method
ORANGE = "#eb6834"    # exposure contribution
GREY = "#b8b7b2"      # reference levels (start/end)

# --- data (our own, dose-matched, 5 seeds) ---
BASELINE = 92.64      # 05 untrained baseline (wikitext sham 02 confirms it: 93.76%)
EXPOSURE = 17.60      # 03 scenario-exposure (no SOO objective)
SOO = 1.60            # 01 full SOO
EXP_DROP = BASELINE - EXPOSURE   # 75.04 pp
SOO_DROP = EXPOSURE - SOO        # 16.00 pp

fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

x = [0, 1, 2, 3]
labels = [
    "Untrained baseline\n(05)",
    "+ Scenario exposure\n(03 · no SOO objective)",
    "+ SOO objective\n(01 · full SOO)",
    "Final: SOO\n(01)",
]
W = 0.62

SHAM = 93.76  # 02 wikitext sham — a control confirming generic FT stays at baseline

# 1) baseline full bar
ax.bar(x[0], BASELINE, W, color=GREY, zorder=3)
# 2) exposure decrement (floats from EXPOSURE up to BASELINE)
ax.bar(x[1], EXP_DROP, W, bottom=EXPOSURE, color=ORANGE, zorder=3)
# 3) SOO decrement (floats from SOO up to EXPOSURE)
ax.bar(x[2], SOO_DROP, W, bottom=SOO, color=BLUE, zorder=3)
# 4) final full bar
ax.bar(x[3], SOO, W, color=GREY, zorder=3)

# 02 wikitext sham: a reference marker showing generic FT lands at baseline.
# Drawn as a hatched cap over the baseline bar so it reads as "same level".
ax.plot([x[0] - W / 2, x[0] + W / 2], [SHAM, SHAM], color=INK2, lw=1.6,
        ls=(0, (2, 2)), zorder=5)
ax.annotate("02 · generic fine-tuning (wikitext)\nlands at baseline — 93.8%",
            xy=(x[0] - W / 2, SHAM), xytext=(x[0] - 0.52, 78),
            fontsize=8.5, color=INK2, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

# connector lines between steps
for xi, y in [(0, BASELINE), (1, EXPOSURE), (2, SOO)]:
    ax.plot([xi + W / 2, xi + 1 - W / 2], [y, y], color=INK2, lw=1, ls=(0, (3, 3)), zorder=2)

# --- value / delta labels ---
ax.text(x[0], BASELINE + 2.5, f"{BASELINE:.1f}%", ha="center", va="bottom",
        fontsize=13, fontweight="bold", color=INK)
ax.text(x[3], SOO + 2.5, f"{SOO:.1f}%", ha="center", va="bottom",
        fontsize=13, fontweight="bold", color=INK)
# decrement magnitudes, centered on each floating bar
ax.text(x[1], EXPOSURE + EXP_DROP / 2, f"−{EXP_DROP:.0f} pp", ha="center", va="center",
        fontsize=13, fontweight="bold", color="white", zorder=4)
ax.text(x[2], SOO + SOO_DROP / 2 + 1.5, f"−{SOO_DROP:.0f} pp", ha="center", va="center",
        fontsize=12, fontweight="bold", color="white", zorder=4)
# landing value under the exposure decrement (connector shows the SOO one)
ax.text(x[1], EXPOSURE - 3.5, f"lands at {EXPOSURE:.1f}%", ha="center", va="top",
        fontsize=9.5, color=INK2)

# share-of-effect callouts
ax.text(x[1], BASELINE + 2.5, "82% of the total drop", ha="center", va="bottom",
        fontsize=10.5, color=ORANGE, fontweight="bold")
ax.text(x[2], EXPOSURE + 4.0, "18% of the total drop", ha="center", va="bottom",
        fontsize=10.5, color=BLUE, fontweight="bold")

# --- axes styling (recessive) ---
ax.set_ylim(0, 108)
ax.set_xlim(-0.6, 3.6)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10, color=INK)
ax.set_ylabel("Deceptive response rate (%)", fontsize=11, color=INK2)
ax.tick_params(axis="y", labelcolor=INK2, labelsize=9.5, length=0)
ax.tick_params(axis="x", length=0)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(INK2)
ax.yaxis.grid(True, color="#e6e5e1", lw=0.8, zorder=0)
ax.set_axisbelow(True)

# --- title / subtitle / source ---
fig.suptitle("Most of the deception drop is scenario exposure, not the SOO objective",
             x=0.02, y=0.98, ha="left", fontsize=14.5, fontweight="bold", color=INK)
ax.set_title("Mistral-7B reproduction · dose-matched control ladder · deceptive response rate",
             loc="left", fontsize=10.5, color=INK2, pad=26)

legend = [
    Patch(facecolor=ORANGE, label="Scenario exposure (data-driven)"),
    Patch(facecolor=BLUE, label="SOO objective (mechanism-specific)"),
    Patch(facecolor=GREY, label="Measured level (baseline / final)"),
]
ax.legend(handles=legend, loc="upper right", frameon=False, fontsize=9.5,
          labelcolor=INK)

fig.text(0.02, 0.015,
         "Our own numbers, 5 seeds, Sonnet-judged. 05 baseline: 92.64±1.80%  ·  "
         "03: 17.60±11.41%  ·  01: 1.60±1.57%.  Wikitext sham (02) confirms the "
         "baseline: 93.76% — generic fine-tuning changes nothing.  03's high "
         "variance makes the 82/18 split approximate; the decomposition is robust.",
         fontsize=7.6, color=INK2, ha="left")

plt.tight_layout(rect=[0, 0.04, 1, 0.94])
out = __file__.rsplit("make_waterfall.py", 1)[0] + "control_ladder_waterfall.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print(f"wrote {out}")
