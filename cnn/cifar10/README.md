# CIFAR-10 CNN

A CNN trained on CIFAR-10, built up incrementally: first to understand each architectural piece (conv output dimensions, why pooling is needed, why activations matter for conv layers too, not just FC), then extended step by step — per-epoch train/val tracking, data augmentation, batch norm, best-checkpoint restoration, patience-based early stopping, hyperparameter search, and confusion-matrix/misclassified-image analysis. Each addition was driven by a concrete problem observed in the previous run, not added speculatively — see **Results timeline** below for the full story of what changed and why, in order.

**Bottom line: test accuracy went from 73.86% → 82.43% over the course of this, entirely through training methodology and hyperparameters — the architecture itself never got deeper.**

## Files

| File | Purpose |
|---|---|
| `cifar_data.py` | `CustomCIFAR10` (train/val/test split via `torch.utils.data.random_split`) + `build_dataloaders()` + `visualize_image()`. Separate `TRAIN_TRANSFORM` (with augmentation) and `EVAL_TRANSFORM` (clean) — augmentation must never touch val/test data. |
| `cifar_model.py` | `CNNcifar10` — 3×(conv+batchnorm+relu+maxpool) → flatten → 3×FC with dropout. `dropout_rate` is a constructor param. |
| `train_engine.py` | `train_one_epoch`, `evaluate`, `fit` (per-epoch history, best-checkpoint restore, patience-based early stopping), `save_checkpoint`, `load_checkpoint` — all device-parameterized, no globals. |
| `cnn_cifar10.py` | Baseline training run: builds data/model, trains, evaluates on the held-out test set, saves a checkpoint, plots the train-vs-val loss/accuracy curve. |
| `hyperparam_search.py` | Random search over learning rate / batch size / dropout rate, logs every trial to `results/hparam_search_<timestamp>.csv`, retrains and checkpoints the winning config. |
| `visualize_results.py` | Loads a checkpoint (no retraining) and produces a confusion matrix + a misclassified-image gallery. |

## Architecture

Input `32×32×3` → `conv(3→32,k3,p1) → batchnorm → relu → maxpool(2)` → `16×16×32` → `conv(32→64,k3,p1) → batchnorm → relu → maxpool(2)` → `8×8×64` → `conv(64→128,k3,p1) → batchnorm → relu → maxpool(2)` → `4×4×128` → flatten (`2048`) → `FC(2048→512) → relu → dropout → FC(512→128) → relu → dropout → FC(128→10)`.

`nn.BatchNorm2d(num_features)` sits between each conv and its activation — `num_features` matches that conv's *output* channels (32/64/128). Normalizes each channel's activations to ~zero mean/unit variance (plus a small learnable scale+shift), which stabilizes training and lets the model train more effectively over more epochs.

## How `fit()` works (the two key mechanisms)

**Best-checkpoint restoration.** The final epoch of a training run is not necessarily the *best* epoch — a model can start overfitting before training stops. `fit()` tracks `best_val_loss` across epochs; every time a new epoch beats it, it stores `copy.deepcopy(model.state_dict())` (a real independent copy — `state_dict()` alone returns references to live tensors that keep mutating as training continues). After the loop, it calls `model.load_state_dict(best_state_dict)` — so whatever the caller does next (evaluate, save a checkpoint) uses the *best* epoch's weights, not whatever the last epoch happened to leave behind.

**Patience-based early stopping.** Rather than guessing a fixed epoch count, `fit(..., num_epochs=X, patience=N)` trains for *up to* `X` epochs but stops early if `N` consecutive epochs pass without a new best `val_loss`. Important, non-obvious detail: patience compares each epoch against the **all-time best**, not the epoch immediately before it. This is intentional and matches how every major ML framework defines it (Keras, PyTorch Lightning) — comparing only to the previous epoch would actually be *more* noise-sensitive (any single bad epoch right after a good one would look like "no improvement" and reset nothing useful); comparing to the all-time best gives the model a full `patience`-epoch window, starting from its best moment, to set a new record however it gets there. The real tradeoff: if the "best" epoch was itself a lucky noisy outlier, training can stop a bit before it truly should — an accepted risk of early stopping in general, not specific to this implementation.

`num_epochs` now means "hard ceiling" rather than "exact training length" — patience is what actually decides when training stops in the normal case.

## Bugs & lessons found along the way

- **`device` read from a module-level global** instead of being passed as a parameter — silently prevented ever comparing CPU vs GPU runs in the same process. Fixed by making `device` a required argument throughout `train_engine.py`.
- **`train_frac=0.8` parameter declared but never used** in the original `CustomCIFAR10` — dead code that turned out to be exactly what was needed for a real train/val split. Wired up via `torch.utils.data.random_split` with a fixed seed (the same seed + same split sizes must be used for both the `train` and `val` instances, or the two splits silently stop being complementary).
- **Training ran at module import time**, no `if __name__ == '__main__':` guard — meant importing the model/data code for reuse (e.g. from a hyperparameter search script) would trigger a full training run as a side effect.
- **Test accuracy only checked once, after all epochs finished** — no visibility into *where* overfitting starts. Fixed by adding a validation pass every epoch and tracking history.
- Earlier, while first writing the training loop: missing `optimizer.step()` (gradients computed but never applied — model literally never learned), comparing raw logits to labels directly instead of `argmax`-ing first (crashes on shape mismatch), and accumulating the loss *tensor* instead of `loss.item()` (keeps the whole computation graph alive for the epoch — a real memory leak).
- **`save_checkpoint`'s metadata was a free-form dict** (`save_checkpoint(model, path, metadata={...})`) — nothing in the signature told you it *must* contain `dropout_rate`, or `load_checkpoint` would `KeyError` later, far from the actual mistake. Redesigned to `save_checkpoint(model, path, dropout_rate, **extra_metadata)` — `dropout_rate` explicit and required (a `TypeError` at the call site if forgotten, not a confusing failure later), everything else flows into `**extra_metadata` freely. **Then made the same mistake twice while updating call sites** — passing `metadata={...}`, then later `extra_metadata={...}`, as a literal keyword — both times this doesn't crash, it just nests the whole dict one level too deep under that literal key name, silently. `**kwargs` catches *unmatched* keyword names; passing a keyword that happens to share a name with the catch-all variable doesn't unpack it. Caught both times by tracing the exact call signature rather than assuming a fix that "runs without error" is correct.
- **Early-stopping `break` was placed before the `history[...].append(...)` call** — meant the exact epoch that triggered the stop (the evidence for why it stopped) was silently missing from the returned history and never printed, even though it cost real compute to run. Fixed by recording/printing every epoch that runs, *then* checking whether to break.
- **Pylance/type-checking recurring pattern**: several `reportOptionalMemberAccess`/`reportAttributeAccessIssue` warnings turned out to be the same shape of issue — a value's static type is a union (e.g. `nn.Conv2d.bias: Optional[Parameter]`, or `CustomCIFAR10.dataset: CIFAR10 | Subset[Unknown]`) and the type checker can't prove which branch applies. Where the branch *is* knowable at that point in the code, `assert isinstance(x, SpecificType)` both narrows the type **and** doubles as a real runtime guard. Where it's genuinely not knowable (inside `__getitem__`, before you know which branch built `self.dataset`), a `# type: ignore` comment is the honest choice — cosmetic-only, doesn't change runtime behavior.

## Results timeline

Each row changed exactly one thing (or was a deliberate, isolated experiment) from the row above, so the accuracy delta can actually be attributed to that change:

| Stage | Change | Test Accuracy |
|---|---|---|
| 0 | Original single-file script, 5 epochs, no val split | 76.36% |
| 1 | Reorganized + real 40k/10k train/val split, per-epoch tracking (no aug/batchnorm yet) | 73.86% (drop is expected — less training data now that 20% is held out for validation) |
| 2 | + data augmentation (`RandomCrop(32,padding=4)` + `RandomHorizontalFlip`) + `BatchNorm2d`, 10 epochs, **no best-checkpoint restore yet** | 76.49% (used epoch 10's weights — but epoch 9 was actually better, see below) |
| 3 | + best-checkpoint restoration in `fit()` (same run, same hyperparameters) | 77.68% (now correctly using epoch 9's weights) |
| 3b | *Experiment*: added `RandomRotation(10)` on top of stage 3 | 75.77% — **made it worse**, reverted (see below) |
| 4 | + first hyperparameter search (fixed epoch budgets: 6 for ranking, 12 for final retrain, no patience yet) | 78.85% — but every trial hit its epoch ceiling without plateauing (see below) |
| 5 | + patience-based early stopping in `fit()`, re-ran hyperparameter search with a proper epoch ceiling | **82.43%** (final) |

**Stage 2→3, why it matters:** stage 2's run showed val accuracy peak at epoch 9 (77.58%) then drop at epoch 10 (76.67%), while train accuracy kept climbing — textbook overfitting starting right at the end of training. Since nothing captured epoch 9's weights, the saved model was epoch 10's slightly-worse ones. This is what motivated best-checkpoint restoration.

**Stage 3b, the rotation experiment (negative result, kept intentionally):** CIFAR-10 photos are real-world objects, almost always upright — rotation doesn't correspond to a realistic input variation the way translation (crop) or mirroring (flip) do. Adding it made the training task harder (lower train accuracy at every epoch) without a compensating generalization benefit — test accuracy dropped by ~2pts relative to stage 3. Not every plausible-sounding augmentation helps; it's dataset-dependent, and this is why you isolate one variable at a time rather than stacking changes and hoping.

**Stage 4, why the search budget mattered:** all 10 trials hit `best_epoch=6` — the *ceiling* of the 6-epoch search budget — meaning none of them had actually started overfitting within that budget. The search was ranking "who's furthest along at epoch 6," not "who converges best eventually." The final retrain showed the same symptom: its best epoch was 12, again the ceiling. This directly motivated patience-based early stopping over guessing ever-larger fixed epoch counts.

**Stage 5, the winning configuration:** `lr=0.00066`, `batch_size=32`, `dropout_rate=0.466`. Final retrain ran to epoch 37 (ceiling was 40) before 5 consecutive epochs with no improvement triggered the stop; best epoch was **32** (`val_loss=0.5007`, `val_acc=83.01%` — both metrics agreeing, a reliable signal). Test accuracy: **82.43%**.

## Hyperparameter search — first run (historical reference)

Random search (not grid — with 3 params at even 3 values each, grid is 27 full runs; random search covers the space far cheaper, since learning rate dominates CNN training dynamics much more than dropout does). This was the *first* search, before patience-based early stopping existed, using fixed epoch budgets:

| Trial | lr | batch_size | dropout | best_epoch | val_loss | val_acc |
|---|---|---|---|---|---|---|
| 0 | 0.00048 | 128 | 0.204 | 3 | 1.1254 | 59.74% |
| 1 | 0.00042 | 32 | 0.242 | 3 | 0.9468 | 66.10% |
| 2 | 0.00203 | 64 | 0.283 | 3 | 0.9622 | 66.35% |
| 3 | 0.00850 | 128 | 0.119 | 3 | 1.5542 | 42.78% |
| 4 | 0.00010 | 64 | 0.179 | 3 | 1.3525 | 50.64% |
| 5 | 0.00035 | 64 | 0.285 | 3 | 1.0568 | 62.09% |
| 6 | 0.00320 | 128 | 0.220 | 3 | 0.9494 | 66.40% |
| 7 | 0.00968 | 128 | 0.396 | 2 | 2.3032 | 9.67% |
| 8 | 0.00549 | 32 | 0.362 | 3 | 1.5896 | 41.24% |
| 9 (winner) | 0.00053 | 64 | 0.328 | 3 | 0.9298 | 66.60% |

**Trial 7 shows real training instability**: `lr=0.00968` combined with `dropout=0.396` collapsed to 9.67% val accuracy — essentially random guessing across 10 classes. High learning rate + high dropout together pushed the model past the point where it could converge at all within the short trial budget.

This search's winner, retrained without patience for a fixed 12 epochs, reached 78.85% test accuracy (stage 4 above) — beaten by the properly-tuned stage 5 search once patience replaced the fixed epoch guess. Full trial logs for both searches: `results/hparam_search_<timestamp>.csv`.

## Confusion matrix & misclassified images

Generated by `visualize_results.py` against the stage 5 checkpoint (82.43% test accuracy). Both PNGs live in `results/` (`confusion_matrix_<run_id>.png`, `misclassified_gallery_<run_id>.png`).

**The confusion matrix's biggest off-diagonal values are all semantically sensible confusions, not random noise:**
- **dog↔cat**: 155 dogs predicted as cats, 98 cats predicted as dogs — by far the largest confusion pair in the whole matrix. Makes sense — both are furry, similarly-posed, similarly-colored animals at 32×32 resolution.
- **automobile↔truck**: 63 trucks predicted as automobiles, 42 automobiles predicted as trucks — visually overlapping vehicle categories.
- **airplane↔ship**: 71 airplanes predicted as ships (the largest single confusion for the airplane row besides correct predictions), 22 ships predicted as airplanes — likely shared context (both often photographed against open sky/water backgrounds) and similar elongated body shapes.

The misclassified gallery (sampled up to 3 per true class, so it's not dominated by whichever single class is worst) backs this up on inspection: mistakes are visually understandable, not bizarre — a big animal filling the frame, low contrast against the background, or a shape genuinely ambiguous at this resolution. This is a healthy sign: the model has learned real, semantically meaningful structure rather than memorizing arbitrary pixel patterns — its errors look like the kind of mistakes a human squinting at a small blurry photo might also make.

## Possible follow-ups

- `build_dataloaders` loads the official CIFAR-10 training set from disk twice (once each for the `train` and `val` `CustomCIFAR10` instances) — correct, but wastes memory; could share one loaded base dataset between both `Subset`s instead.
- A proper hyperparameter search would use the search budget more intelligently (e.g. successive halving/Hyperband — start all trials cheap, keep only the top fraction, give survivors more epochs) rather than a flat epoch cutoff for every trial.
- Try wiring `MyConv2d` (from `../my_conv2d/`) into this model in place of `nn.Conv2d`, now that both the from-scratch conv layer and this training pipeline exist independently.
- Given how much methodology alone moved accuracy (73.86% → 82.43%), a deeper architecture (more conv layers, or residual connections) is the natural next lever now that the training process itself is solid.
- `hyperparam_search.py`'s winning-config retrain doesn't call `plot_history` the way `cnn_cifar10.py` does — there's no saved train-vs-val curve for the actual 82.43% run, only for earlier `cnn_cifar10.py` baseline runs. Worth adding, especially now that `fit()` returns full per-epoch history including whichever epoch patience stopped at.
