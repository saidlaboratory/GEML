import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# EXACT measured Phase-B value-head numbers (results/PHASE_B_GPU.md, DETAILED_RESULTS.md)
seeds = ["26", "27", "28"]
val_mae   = [0.8472, 0.8376, 0.8296]
iid_mae   = [1.9162, 1.8887, 1.8685]
ood_mae   = [1.9076, 1.8804, 1.8600]     # constant predictor on test-OOD
iid_spear = [-0.0142, 0.0021, -0.0089]   # test-IID Spearman ~ 0
# test-OOD Spearman: undefined (constant predictor)

plt.rcParams.update({
    "font.size": 8, "font.family": "serif",
    "axes.linewidth": 0.6, "axes.edgecolor": "#444",
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.35, 1.75))
x = np.arange(len(seeds)); w = 0.38
c_err, c_rank = "#3b6fb0", "#b03b3b"

# Panel A: regression error looks fine -- even the constant predictor's
w = 0.26
ax1.bar(x - w, val_mae, w, color=c_err, label="val")
ax1.bar(x, iid_mae, w, color=c_err, alpha=0.5, label="test-IID")
ax1.bar(x + w, ood_mae, w, facecolor="none", edgecolor=c_err, lw=0.7,
        hatch="///", label="OOD (constant)")
ax1.set_title("Regression error (MAE)", fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels(seeds); ax1.set_xlabel("seed")
ax1.set_ylim(0, 3.0); ax1.legend(frameon=False, fontsize=5.5, loc="upper left")
ax1.spines[["top","right"]].set_visible(False)

# Panel B: ranking is zero
ax2.axhspan(-0.05, 0.05, color="#bbb", alpha=0.35, lw=0)
ax2.axhline(0, color="#444", lw=0.6)
ax2.bar(x, iid_spear, 0.5, color=c_rank)
ax2.set_title("Rank correlation (Spearman)", fontsize=8)
ax2.set_xticks(x); ax2.set_xticklabels(seeds); ax2.set_xlabel("seed")
ax2.set_ylim(-0.6, 0.6)
ax2.text(0.5, 0.42, "test-IID $\\approx$ chance", ha="center", fontsize=6, color=c_rank)
ax2.text(0.5, -0.5, "test-OOD: constant\n(Spearman undefined)", ha="center", fontsize=5.5, color="#666")
ax2.spines[["top","right"]].set_visible(False)

fig.tight_layout(pad=0.4)
fig.savefig("fig_valuehead.pdf", bbox_inches="tight")
fig.savefig("fig_valuehead.png", dpi=300, bbox_inches="tight")
print("wrote fig_valuehead.pdf and fig_valuehead.png")
