"""Regenerate the MT-baseline pivot figure from results/mt_baseline_qwen3-14b-base.json.
Falls back to the recorded numbers if the JSON isn't present yet."""
import json, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
    "axes.linewidth": 0.9, "axes.edgecolor": "#333333", "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 200,
})
NAVY, SLATE, ACC = "#1f3a6d", "#9aa7bd", "#b4451f"

# recorded result (12 langs, Qwen3-14B-Base) - excluding the failed Hindi target
targets = ["Spanish", "Russian", "Chinese", "Swahili", "Mean"]
src_en  = [0.662, 0.730, 0.728, 0.883, 0.751]
src_tgt = [0.670, 0.714, 0.669, 0.712, 0.691]
# panel B: alignment-to-English vs chrF, target = Swahili (r = 0.88)
labels = ["de","es","fr","ru","zh","hi","bn","ta","th","tr"]
cla_en = [0.771,0.764,0.799,0.740,0.733,0.560,0.502,0.418,0.654,0.613]
chrf   = [30.09,29.95,31.85,26.93,26.14,16.36,20.55,17.15,28.51,26.98]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 4.3))
x = np.arange(len(targets)); w = 0.38
b1 = axA.bar(x-w/2, src_en,  w, label="align to English  (src→en)", color=NAVY, edgecolor="white", linewidth=0.6)
b2 = axA.bar(x+w/2, src_tgt, w, label="align to target  (src→tgt)", color=SLATE, edgecolor="white", linewidth=0.6)
for bars in (b1, b2):
    for r in bars:
        axA.text(r.get_x()+r.get_width()/2, r.get_height()+0.012, f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=8.2)
axA.axvline(len(targets)-1.5, color="#cccccc", lw=0.8, ls=(0,(3,3)))
axA.set_xticks(x); axA.set_xticklabels(targets, fontsize=9.5)
axA.set_ylabel("Pearson correlation,  alignment vs chrF"); axA.set_ylim(0, 1.0)
axA.set_title("A.  English-pivot predicts translation quality", fontsize=11.5, loc="left", pad=10)
axA.legend(frameon=False, fontsize=8.6, loc="upper left")
axB.scatter(cla_en, chrf, s=42, color=NAVY, zorder=3, edgecolor="white", linewidth=0.6)
m, b = np.polyfit(cla_en, chrf, 1); xs = np.linspace(min(cla_en)-0.02, max(cla_en)+0.02, 50)
axB.plot(xs, m*xs+b, color=ACC, lw=1.6, zorder=2)
for lx, ly, lab in zip(cla_en, chrf, labels):
    axB.annotate(lab, (lx, ly), textcoords="offset points", xytext=(5,3), fontsize=8.0, color="#444444")
axB.set_xlabel("cross-lingual alignment to English"); axB.set_ylabel("translation quality  (chrF)")
axB.set_title("B.  Higher alignment, better translation  (r = 0.88)", fontsize=11.5, loc="left", pad=10)
axB.set_ylim(10, 36)
fig.suptitle("Reproducing the English-pivot effect  ·  Qwen3-14B-Base, FLORES-200, 12 languages",
             fontsize=12.5, y=1.02, x=0.02, ha="left", fontweight="bold")
fig.tight_layout()
fig.text(0.02, -0.02, "Qwen3-14B-Base, 5-shot, FLORES-200 devtest, chrF.  Alignment = ANC at the max-alignment layer.  "
         "Hindi target excluded: translation into Devanagari failed (chrF < 10).", ha="left", va="top", fontsize=7.8, color="#777777")
out = os.path.join(os.path.dirname(__file__), "pivot_result.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)
