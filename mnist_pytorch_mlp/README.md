# Custom MLP for MNIST (PyTorch)

A multi-layer perceptron built on top of `nn.Module`, trained on MNIST, with a focus on understanding what's actually happening during training rather than just calling `.fit()`.

## What's in here

- [`custom_mlp_pytorch.py`](custom_mlp_pytorch.py) — the full script, written as a step-by-step walkthrough (see the numbered comments inside):
  - **Steps 0–3**: a configurable `CustomMLP` (arbitrary hidden layers, relu/tanh activation, classification/regression output), MNIST loading, loss/optimizer setup.
  - **Steps 4–5**: a single manual training step, then wrapped in an epoch loop, to see loss/accuracy change by hand before reaching for a reusable training function.
  - **Step 6**: a reusable `train_one_config()` function, used to compare **Full-batch vs Mini-batch (batch_size=64) vs Stochastic (batch_size=1) Gradient Descent** on identical architecture/optimizer/data — tracking loss, accuracy, and wall-clock time per epoch, plus per-epoch validation metrics to check for overfitting.
- [`gd_comparison.png`](gd_comparison.png) — loss & accuracy for all three configs, plotted against both epoch and wall-clock time (log-scaled on both axes — the three configs run wildly different numbers of epochs and take wildly different amounts of time, so a linear scale makes the faster runs unreadable).
- [`gd_overfitting_check.png`](gd_overfitting_check.png) — train vs. validation curves per config, with each config's best (lowest validation loss) epoch marked — the overfitting analysis.
- [`gd_comparison_results.json`](gd_comparison_results.json) — cached training results (loss/accuracy/time history per config). The script loads this instead of retraining if it already exists; delete it to force a fresh run.

## Bugs found along the way

Two are worth calling out, since neither crashed anything — they just quietly produced wrong numbers:

1. **Double softmax.** The model applied `F.softmax` on the output layer for multiclass classification, then fed that into `nn.CrossEntropyLoss`, which already applies `log_softmax` internally. Softmax-ing twice flattens gradients and slows training. Fix: return raw logits from the model; let the loss function handle the softmax.
2. **Loss aggregation bug.** Per-epoch average loss was computed as `loss_batch.item() / batch_size`, accumulated across all batches — but `CrossEntropyLoss` already averages within a batch by default, so dividing again double-shrinks it. The bug happened to look plausible for one batch size and wildly wrong for another (stochastic GD's reported loss came out ~30x too high) purely from bookkeeping, not an actual training problem. Fix: accumulate each batch's *total* loss (`loss.item() * batch_size`), divide by the total sample count once at the end of the epoch.

## What the experiment found

- **Mini-batch converges fastest**, in both epoch count and wall-clock time.
- **Full-batch is slow but stable** — no sign of overfitting even after 40 epochs; train and validation curves track each other closely the whole way.
- **Mini-batch starts overfitting around epoch 7** — validation loss bottoms out there, then rises 68% by epoch 20, while training loss keeps falling toward zero.
- **Stochastic (batch_size=1) overfits almost immediately** — its best validation loss shows up at epoch 1, out of only 5 epochs run, despite making 60,000 weight updates in that single epoch.

Takeaway: more frequent weight updates make a model learn faster — but also make it memorize the training set faster. Same lever drives both effects.

## Reproducing

```
python custom_mlp_pytorch.py
```

Requires `torch`, `keras` (used only for MNIST loading / one-hot encoding), and `matplotlib`. The first run trains all three configs (the stochastic run takes several minutes on CPU); later runs load cached results from `gd_comparison_results.json` unless you delete it.
