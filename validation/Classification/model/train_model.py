import os
import torch
import torch.nn as nn
import numpy as np

from plots import plot_accuracy

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, dl, device):
    """Stroke probability and true label for every clip in a data loader (eval mode)."""
    model.eval(); probs, ys = [], []
    with torch.no_grad():
        for x, m, y in dl:
            x, m = x.to(device), m.to(device)
            probs.append(model(x, m).softmax(1)[:, 1].cpu()); ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def sensitivity_specificity(probs, labels, threshold=0.5):
    """Sensitivity = stroke clips called stroke (TPR); specificity = healthy clips called healthy (TNR)."""
    pred = (probs >= threshold).astype(int)
    return float((pred[labels == 1] == 1).mean()), float((pred[labels == 0] == 0).mean())


def balanced_accuracy(probs, labels, threshold=0.5):
    return float(np.mean(sensitivity_specificity(probs, labels, threshold)))


METRIC_COLUMNS = ["split", "fold", "level", "n", "n_stroke", "sensitivity", "specificity",
                  "balanced_acc", "auc", "tp", "fn", "tn", "fp"]


def metrics_row(split, fold, level, probs, labels, threshold=0.5):
    """One row of the metrics table (a dict keyed by METRIC_COLUMNS)."""
    from sklearn.metrics import roc_auc_score
    pred = (probs >= threshold).astype(int)
    sens, spec = sensitivity_specificity(probs, labels, threshold)
    return {"split": split, "fold": fold, "level": level, "n": len(labels), "n_stroke": int(labels.sum()),
            "sensitivity": sens, "specificity": spec, "balanced_acc": (sens + spec) / 2,
            "auc": float(roc_auc_score(labels, probs)),
            "tp": int(((pred == 1) & (labels == 1)).sum()), "fn": int(((pred == 0) & (labels == 1)).sum()),
            "tn": int(((pred == 0) & (labels == 0)).sum()), "fp": int(((pred == 1) & (labels == 0)).sum())}


def per_participant(probs, labels, participant_ids):
    """Average each participant's clip probabilities -> (probs, labels), one per participant."""
    ids = np.asarray(participant_ids); u = np.unique(ids)
    return (np.array([probs[ids == i].mean() for i in u]),
            np.array([labels[ids == i][0] for i in u]))


def summary_rows(rows, split):
    """Mean and SD rows over folds for the numeric metric columns."""
    keys = ["sensitivity", "specificity", "balanced_acc", "auc"]
    out = []
    for name, fn in [("mean", np.mean), ("sd", np.std)]:
        r = {c: "" for c in METRIC_COLUMNS}
        r.update(split=split, fold=name, level=rows[0]["level"],
                 **{k: float(fn([row[k] for row in rows])) for k in keys})
        out.append(r)
    return out


def save_metrics_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def training_loop(model, train_dl, train_labels, val_dl,
                  epochs=50, lr=3e-4, warmup_frac=0.05, out_dir="results"):
    """Train, keep the epoch with the best validation balanced accuracy.
    Returns the path of the saved best model."""
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

    train_acc, val_acc, best_acc = [], [], -1.0
    for epoch in range(epochs):
        model.train()
        for x, m, y in train_dl:
            x, m, y = x.to(device), m.to(device), y.to(device)
            loss = loss_fn(model(x, m), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        train_acc.append(balanced_accuracy(*predict(model, train_dl, device)))
        val_acc.append(balanced_accuracy(*predict(model, val_dl, device)))
        if val_acc[-1] > best_acc:
            best_acc = float(val_acc[-1])          # plain float so torch.load(weights_only) accepts it
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_bal_acc": best_acc}, best_path)
        print(f"  epoch {epoch:3d}  lr {sched.get_last_lr()[0]:.2e}  loss {loss.item():.3f}  "
              f"train acc {train_acc[-1]:.3f}  val acc {val_acc[-1]:.3f}")

    plot_accuracy(train_acc, val_acc, os.path.join(out_dir, "accuracy.png"))
    print(f"  best val balanced acc {best_acc:.3f}; model saved to {best_path}")
    return best_path
