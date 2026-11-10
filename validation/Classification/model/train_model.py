import os
import torch
import torch.nn as nn
import numpy as np

from plots import plot_accuracy, plot_loss

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(t, device):
    """A tensor, or a tuple of tensors (one per stream), moved to device."""
    return tuple(a.to(device) for a in t) if isinstance(t, (list, tuple)) else t.to(device)


def forward(model, x, m):
    """Transformer: model(x, mask); CA_Transformer: model(motion, emg, motion_mask, emg_mask)."""
    return model(*x, *m) if isinstance(x, (list, tuple)) else model(x, m)


def predict(model, dl, device):
    """Stroke probability and true label for every clip in a data loader (eval mode)."""
    model.eval(); probs, ys = [], []
    with torch.no_grad():
        for x, m, y in dl:
            x, m = to_device(x, device), to_device(m, device)
            probs.append(forward(model, x, m).softmax(1)[:, 1].cpu()); ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def sensitivity_specificity(probs, labels, threshold=0.5):
    """Sensitivity = stroke clips called stroke (TPR); specificity = healthy clips called healthy (TNR)."""
    pred = (probs >= threshold).astype(int)
    return float((pred[labels == 1] == 1).mean()), float((pred[labels == 0] == 0).mean())


def balanced_accuracy(probs, labels, threshold=0.5):
    return float(np.mean(sensitivity_specificity(probs, labels, threshold)))


METRIC_COLUMNS = ["split", "model", "level", "n", "n_stroke", "sensitivity", "specificity",
                  "balanced_acc", "auc", "tp", "fn", "tn", "fp"]


def metrics_row(split, model, level, probs, labels, threshold=0.5):
    """One row of the metrics table (a dict keyed by METRIC_COLUMNS)."""
    from sklearn.metrics import roc_auc_score
    pred = (probs >= threshold).astype(int)
    sens, spec = sensitivity_specificity(probs, labels, threshold)
    return {"split": split, "model": model, "level": level, "n": len(labels), "n_stroke": int(labels.sum()),
            "sensitivity": sens, "specificity": spec, "balanced_acc": (sens + spec) / 2,
            "auc": float(roc_auc_score(labels, probs)),
            "tp": int(((pred == 1) & (labels == 1)).sum()), "fn": int(((pred == 0) & (labels == 1)).sum()),
            "tn": int(((pred == 0) & (labels == 0)).sum()), "fp": int(((pred == 1) & (labels == 0)).sum())}


def per_participant(probs, labels, participant_ids):
    """Average each participant's clip probabilities -> (probs, labels), one per participant."""
    ids = np.asarray(participant_ids); u = np.unique(ids)
    return (np.array([probs[ids == i].mean() for i in u]),
            np.array([labels[ids == i][0] for i in u]))


def save_metrics_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def mean_loss(model, dl, loss_fn, device):
    """Average loss over a data loader in eval mode (weighted by batch size)."""
    model.eval(); total, n = 0.0, 0
    with torch.no_grad():
        for x, m, y in dl:
            x, m, y = to_device(x, device), to_device(m, device), y.to(device)
            total += loss_fn(forward(model, x, m), y).item() * len(y); n += len(y)
    return total / n


def training_loop(model, train_dl, train_labels, val_dl,
                  epochs=50, lr=3e-4, warmup_frac=0.05, out_dir="results", extra=None):
    """Train, keep the epoch with the best validation balanced accuracy.
    `extra` is stored in the checkpoint (e.g. hyperparameters, normalisation stats).
    Writes accuracy.png and loss.png to out_dir. Returns (best model path, history dict)."""
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best_model.pt")
    device = get_device()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    # Warmup then cosine decay; stepped once per batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_dl),
                                                pct_start=warmup_frac)

    w = torch.tensor(len(train_labels) / (2 * np.bincount(train_labels)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w.to(device))          # handles class imbalance

    hist = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_acc, best_epoch = -1.0, 0
    for epoch in range(epochs):
        model.train(); total, n = 0.0, 0
        for x, m, y in train_dl:
            x, m, y = to_device(x, device), to_device(m, device), y.to(device)
            loss = loss_fn(forward(model, x, m), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += loss.item() * len(y); n += len(y)

        hist["train_loss"].append(total / n)                    # mean over the epoch's batches (train mode)
        hist["val_loss"].append(mean_loss(model, val_dl, loss_fn, device))
        hist["train_acc"].append(balanced_accuracy(*predict(model, train_dl, device)))
        hist["val_acc"].append(balanced_accuracy(*predict(model, val_dl, device)))
        if hist["val_acc"][-1] > best_acc:
            best_acc, best_epoch = float(hist["val_acc"][-1]), epoch   # plain float for torch.load(weights_only)
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_bal_acc": best_acc,
                        **(extra or {})}, best_path)
        print(f"  epoch {epoch:3d}  lr {sched.get_last_lr()[0]:.2e}  "
              f"loss train {hist['train_loss'][-1]:.3f} val {hist['val_loss'][-1]:.3f}  "
              f"bal acc train {hist['train_acc'][-1]:.3f} val {hist['val_acc'][-1]:.3f}")

    plot_accuracy(hist["train_acc"], hist["val_acc"], os.path.join(out_dir, "accuracy.png"))
    plot_loss(hist["train_loss"], hist["val_loss"], os.path.join(out_dir, "loss.png"), best_epoch)
    hist["best_epoch"] = best_epoch
    print(f"  best val balanced acc {best_acc:.3f} (epoch {best_epoch + 1}); model saved to {best_path}")
    return best_path, hist
