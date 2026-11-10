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


def plot_roc(fpr, tpr, auc, path, title="ROC"):
    """One ROC curve with the chance diagonal."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, label="Chance")
    ax.plot(fpr, tpr, color=BLUE, lw=2.5, label=f"Model (AUC {auc:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (stroke)")
    ax.set_title(title, fontsize=11)
    _style(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_loss(train_loss, val_loss, path, best_epoch=None):
    """Train vs validation loss per epoch; the saved (best validation accuracy) epoch is marked."""
    epochs = range(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, color=BLUE, lw=2, label="Train")
    ax.plot(epochs, val_loss, color=ORANGE, lw=2, label="Validation")
    if best_epoch is not None:
        ax.axvline(best_epoch + 1, color=MUTED, lw=1, ls="--", label=f"Saved epoch ({best_epoch + 1})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Class-weighted cross-entropy"); ax.set_ylim(bottom=0)
    _style(ax); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
