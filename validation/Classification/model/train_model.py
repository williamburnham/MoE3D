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


def balanced_accuracy(probs, labels, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    return 0.5 * ((pred[labels == 0] == 0).mean() + (pred[labels == 1] == 1).mean())


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
