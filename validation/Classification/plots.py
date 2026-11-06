import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e5e5e5"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def _style(ax):
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_accuracy(train_acc, val_acc, path):
    """Train vs validation balanced accuracy per epoch, best validation epoch marked."""
    epochs = range(1, len(train_acc) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_acc, color=BLUE, lw=2, label="Train")
    ax.plot(epochs, val_acc, color=ORANGE, lw=2, label="Validation")
    best = int(np.argmax(val_acc))
    ax.plot(best + 1, val_acc[best], "o", ms=8, color=ORANGE, mec="white", mew=2)
    ax.annotate(f"best val {val_acc[best]:.3f} (epoch {best + 1})", (best + 1, val_acc[best]),
                xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Balanced accuracy"); ax.set_ylim(0, 1.05)
    _style(ax); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(cm, class_names, path, title="Confusion matrix"):
    """Counts with row-normalised percentages (row = true class)."""
    cm = np.asarray(cm)
    row_frac = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(row_frac, cmap=BLUES, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{100 * row_frac[i, j]:.1f}%", ha="center", va="center",
                    fontsize=11, color="white" if row_frac[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of true class"); cbar.outline.set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc(curves, path, title="ROC", main=None):
    """curves: list of (fpr, tpr, auc), one per fold model, drawn thinly.
    main=None: also draw their mean ROC (TPR interpolated on a common FPR grid) with a +/-1 SD band.
    main=(fpr, tpr, auc): draw that curve bold instead (e.g. the ensemble on the test set)."""
    aucs = np.array([a for _, _, a in curves])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, label="Chance")
    for k, (fpr, tpr, _) in enumerate(curves):
        ax.plot(fpr, tpr, color=BLUE, lw=1, alpha=0.35,
                label=f"Fold models (AUC {aucs.min():.3f}-{aucs.max():.3f})" if k == 0 else None)
    if main is None:
        grid = np.linspace(0, 1, 201)
        tprs = np.array([np.interp(grid, fpr, tpr) for fpr, tpr, _ in curves])
        tprs[:, 0] = 0.0
        mean_tpr, sd_tpr = tprs.mean(0), tprs.std(0)
        mean_tpr[-1] = 1.0
        ax.fill_between(grid, np.clip(mean_tpr - sd_tpr, 0, 1), np.clip(mean_tpr + sd_tpr, 0, 1),
                        color=BLUE, alpha=0.12, lw=0, label="±1 SD")
        ax.plot(grid, mean_tpr, color=BLUE, lw=2.5,
                label=f"Mean ROC (AUC {aucs.mean():.3f} ± {aucs.std():.3f})")
    else:
        fpr, tpr, auc = main
        ax.plot(fpr, tpr, color=BLUE, lw=2.5, label=f"Ensemble (AUC {auc:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (stroke)")
    ax.set_title(title, fontsize=11)
    _style(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
