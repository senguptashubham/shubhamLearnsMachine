import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import matplotlib.pyplot as plt

# reads the already-computed results from run_experiments.py -- no model
# training, no dataloaders, just plotting against saved numbers, so this can
# be re-run freely while iterating on visualization
results_path = os.path.join(os.path.dirname(__file__), "results", "experiment_results.json")
with open(results_path, "r") as f:
  results = json.load(f)

rnn_results = [r for r in results if r["label"] == "RNN"]
x_rnn = [r["T"] for r in rnn_results]
y_rnn = [r["test_accuracy"] for r in rnn_results]

lstm_results = [r for r in results if r["label"] == "LSTM"]
x_lstm = [r["T"] for r in lstm_results]
y_lstm = [r["test_accuracy"] for r in lstm_results]

# same styling tokens as gradient_comparison.py, for a consistent visual
# identity across both plots in this project
SURFACE = "#1a1a19"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRIDLINE = "#2c2c2a"
BASELINE = "#383835"
COLOR_RNN = "#3987e5"    # same blue used for RNN in the gradient plot
COLOR_LSTM = "#d95926"   # same orange used for LSTM in the gradient plot

fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

# marker='o' -- with only 3 points per line, bare lines read as too sparse;
# markers make each actual data point visible, not just the line between them
ax.plot(x_rnn, y_rnn, marker="o", markersize=7, linewidth=2, color=COLOR_RNN, label="RNN")
ax.plot(x_lstm, y_lstm, marker="o", markersize=7, linewidth=2, color=COLOR_LSTM, label="LSTM")

# direct labels on each point -- only 6 points total, cheap to label exactly
for x, y in zip(x_rnn, y_rnn):
  ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10),
              ha="center", color=COLOR_RNN, fontsize=9)
for x, y in zip(x_lstm, y_lstm):
  ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, -14),
              ha="center", color=COLOR_LSTM, fontsize=9)

# recessive chrome: hairline gridlines, no top/right border, muted ticks --
# same recipe as gradient_comparison.py
ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
  ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
  ax.spines[side].set_color(BASELINE)
ax.tick_params(colors=INK_MUTED, labelsize=9)
ax.set_xticks([64, 196, 784])  # force ticks exactly at the 3 tested sequence lengths

ax.set_xlabel("sequence length T", color=INK_SECONDARY, fontsize=10)
ax.set_ylabel("test accuracy (%)", color=INK_SECONDARY, fontsize=10)
ax.set_title("Sequential MNIST: accuracy vs. sequence length, RNN vs LSTM",
             color=INK_PRIMARY, fontsize=13, fontweight="bold", pad=14)
ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=10)

fig.tight_layout()
accuracy_plot_path = os.path.join(os.path.dirname(__file__), "results", "accuracy_vs_T.png")
fig.savefig(accuracy_plot_path, dpi=150, facecolor=SURFACE)
print("saved accuracy plot to:", accuracy_plot_path)

# --- second figure: train vs val LOSS per config, 2x3 grid ---------------
# loss (not accuracy) is the more sensitive diagnostic here -- it's what
# shows instability/NaN spikes most clearly, and train-vs-val loss diverging
# is the classic overfitting signature. Rows = architecture, columns = T,
# same color-per-architecture identity as the first plot (blue=RNN,
# orange=LSTM); train=solid, val=dashed distinguishes the two lines within
# each panel. Each panel keeps its own y-scale (not shared) since loss
# magnitude/dynamics can differ a lot across T, especially if RNN at T=784
# turns out unstable -- sharing axes would compress the well-behaved panels.
row_order = ["RNN", "LSTM"]
col_order = [64, 196, 784]

fig2, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=SURFACE)

for row_idx, label in enumerate(row_order):
  color = COLOR_RNN if label == "RNN" else COLOR_LSTM
  for col_idx, T in enumerate(col_order):
    ax2 = axes[row_idx, col_idx]
    ax2.set_facecolor(SURFACE)
    r = next(r for r in results if r["label"] == label and r["T"] == T)
    epoch_range = range(1, len(r["train_loss_history"]) + 1)

    ax2.plot(epoch_range, r["train_loss_history"], color=color, linewidth=2, linestyle="-", label="train")
    ax2.plot(epoch_range, r["val_loss_history"], color=color, linewidth=2, linestyle="--", label="val")

    ax2.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
      ax2.spines[side].set_visible(False)
    for side in ("left", "bottom"):
      ax2.spines[side].set_color(BASELINE)
    ax2.tick_params(colors=INK_MUTED, labelsize=8)
    ax2.set_title(f"{label}, T={T}", color=INK_PRIMARY, fontsize=11)

    if row_idx == 1:
      ax2.set_xlabel("epoch", color=INK_SECONDARY, fontsize=9)
    if col_idx == 0:
      ax2.set_ylabel("loss", color=INK_SECONDARY, fontsize=9)

# one shared legend explaining the solid/dashed convention, not repeated per panel
handles, labels_ = axes[0, 0].get_legend_handles_labels()
fig2.legend(handles, labels_, loc="upper center", ncol=2, frameon=False,
            labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 1.04))
fig2.suptitle("Train vs. validation loss per config", color=INK_PRIMARY,
              fontsize=14, fontweight="bold", y=1.08)

fig2.tight_layout()
curves_plot_path = os.path.join(os.path.dirname(__file__), "results", "train_val_curves.png")
fig2.savefig(curves_plot_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
print("saved train/val curves plot to:", curves_plot_path)

plt.show()
