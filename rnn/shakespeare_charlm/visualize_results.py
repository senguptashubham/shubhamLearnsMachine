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

rnn = next(r for r in results if r["label"] == "RNN")
lstm = next(r for r in results if r["label"] == "LSTM")

# no T sweep this phase (unlike Phase 2) -- just one RNN run and one LSTM run,
# so the headline comparison is a plain printed stat, not a chart. A bar chart
# of 2 bars would be thin content for a whole figure.
print(f"RNN  | test loss {rnn['test_loss']:.4f}  | test perplexity {rnn['test_perplexity']:.2f}")
print(f"LSTM | test loss {lstm['test_loss']:.4f}  | test perplexity {lstm['test_perplexity']:.2f}")

# same styling tokens as Phase 1/2's plots, for a consistent visual identity
# across the whole project
SURFACE = "#1a1a19"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRIDLINE = "#2c2c2a"
BASELINE = "#383835"
COLOR_RNN = "#3987e5"
COLOR_LSTM = "#d95926"


def plot_train_val_curves(metric_key, ylabel, title, save_name):
  # metric_key: "loss" or "perplexity". Split into TWO panels (train | val)
  # instead of overlaying all 4 lines on one axis -- RNN and LSTM's curves
  # are genuinely close together, and cramming architecture (color) x
  # train/val (linestyle) into one panel made all 4 lines blur into each
  # other. Two lines per panel (RNN vs LSTM, the comparison that actually
  # matters) is far more legible; comparing the two panels side by side still
  # shows the train-vs-val relationship.
  fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=SURFACE)

  for split, ax in [("train", axes[0]), ("val", axes[1])]:
    ax.set_facecolor(SURFACE)
    for result, color, label in [(rnn, COLOR_RNN, "RNN"), (lstm, COLOR_LSTM, "LSTM")]:
      history = result[f"{split}_{metric_key}_history"]
      epoch_range = range(1, len(history) + 1)
      ax.plot(epoch_range, history, color=color, linewidth=2.2, label=label)
      # direct label at the final value -- with curves this close together,
      # exact numbers are easier read as text than traced off the lines
      ax.annotate(f"{history[-1]:.2f}", (epoch_range[-1], history[-1]),
                  textcoords="offset points", xytext=(6, 0), va="center",
                  color=color, fontsize=9, fontweight="bold")
    ax.margins(x=0.12)  # headroom on the right so the end labels don't clip

    # recessive chrome: hairline gridlines, no top/right border, muted ticks --
    # same recipe used throughout this project
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
      ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
      ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.set_xlabel("epoch", color=INK_SECONDARY, fontsize=10)
    ax.set_title(split, color=INK_PRIMARY, fontsize=11)

  axes[0].set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
  # one shared legend (architecture -> color), not repeated per panel
  handles, labels_ = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False,
             labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 1.02), fontsize=10)
  fig.suptitle(title, color=INK_PRIMARY, fontsize=14, fontweight="bold", y=1.1)

  fig.tight_layout()
  plot_path = os.path.join(os.path.dirname(__file__), "results", save_name)
  fig.savefig(plot_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
  print("saved plot to:", plot_path)
  return fig


plot_train_val_curves("loss", "loss", "Shakespeare char-LM: train vs. validation loss", "loss_curves.png")
plot_train_val_curves("perplexity", "perplexity", "Shakespeare char-LM: train vs. validation perplexity", "perplexity_curves.png")

# plt.show() called once, at the very end, after both figures are built and
# saved -- calling it mid-way would block before the second figure exists
plt.show()
