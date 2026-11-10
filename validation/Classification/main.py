import os
import csv
import glob
import json
import shutil
import itertools
from argparse import ArgumentParser
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from torch.utils.data import DataLoader

from dataset.dataset import (CLASSES, MAX_LEN, parse, ClipDataset, train_stats, load_data_config,
                             input_dim, has_modality, check_emg_build)
from model.transformer import Transformer
from model.ca_transformer import CA_Transformer
from model.train_model import (training_loop, predict, get_device, metrics_row, per_participant,
                               save_metrics_csv)
from plots import plot_confusion, plot_roc

GRID_KEYS = ["lr", "dropout", "n_layers", "n_heads"]
D_MODEL = 64


def files_for(files, participants):
    participants = set(participants)
    return [f for f in files if parse(f)[1] in participants]


def labels_for(files):
    return np.array([parse(f)[0] for f in files])


def load_grid(path):
    """Read the hyperparameter config -> list of dicts, one per combination."""
    with open(path) as fh:
        cfg = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    unknown, missing = set(cfg) - set(GRID_KEYS), set(GRID_KEYS) - set(cfg)
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    cfg = {k: v if isinstance(v, list) else [v] for k, v in cfg.items()}
    bad = [h for h in cfg["n_heads"] if D_MODEL % h]
    if bad:
        raise ValueError(f"{path}: n_heads {bad} do not divide d_model={D_MODEL}")
    return [dict(zip(GRID_KEYS, combo)) for combo in itertools.product(*(cfg[k] for k in GRID_KEYS))]


def combo_name(hp):
    return f"lr{hp['lr']:g}_do{hp['dropout']:g}_L{hp['n_layers']}_H{hp['n_heads']}"


def build_model(hp, modality):
    """Transformer for one stream (motion or emg); CA_Transformer (cross-attention) for both."""
    common = dict(d_model=D_MODEL, n_heads=hp["n_heads"], n_layers=hp["n_layers"],
                  dropout=hp["dropout"], max_len=MAX_LEN)
    if modality == "both":
        return CA_Transformer(in_dim_motion=input_dim("motion"), in_dim_emg=input_dim("emg"), **common)
    return Transformer(in_dim=input_dim(modality), **common)


def write_grid_csv(results, path):
    """Grid results, best first (validation balanced accuracy, AUC breaks ties)."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]))
        w.writeheader()
        for r in sorted(results, key=lambda r: (-r["val_balanced_acc"], -r["val_auc"])):
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) and k not in GRID_KEYS else v)
                        for k, v in r.items()})


def make_splits(dataset, modality, test_size, val_size, seed):
    """Participant-level train / val / test split -> (train_files, val_files, test_files).
    Deterministic for a given dataset, modality and seed, so evaluate.py can rebuild it."""
    files = sorted(glob.glob(f"{dataset}/motion/new_joint_vecs/*.npy"))
    files = [f for f in files if parse(f)[0] is not None]          # keep Hlth/Strk only
    n_all = len(files)
    files = [f for f in files if has_modality(f, modality)]        # EMG/both: skip clips with no EMG
    if len(files) < n_all:
        print(f"  {n_all - len(files)} of {n_all} clips have no EMG and are skipped")

    # All splits are over unique participants (not files), so no participant is in two sets
    prt_label = {parse(f)[1]: parse(f)[0] for f in files}
    participants = np.array(sorted(prt_label))
    prt_labels = np.array([prt_label[p] for p in participants])
    print(f"{len(files)} clips, {len(participants)} participants "
          f"({np.bincount(prt_labels).tolist()} per class)")

    # Fixed test set, locked away until the very end; validation set for picking hyperparameters
    prt_rest, prt_test, lab_rest, _ = train_test_split(participants, prt_labels, stratify=prt_labels,
                                                       test_size=test_size, random_state=seed)
    prt_train, prt_val = train_test_split(prt_rest, stratify=lab_rest,
                                          test_size=val_size, random_state=seed)
    train_files, val_files, test_files = (files_for(files, p) for p in (prt_train, prt_val, prt_test))
    print(f"Participants: train {len(prt_train)}, val {len(prt_val)}, test {len(prt_test)}; "
          f"clips: train {len(train_files)}, val {len(val_files)}, test {len(test_files)}")

    return train_files, val_files, test_files


def evaluate_and_save(model, name, val_dl, test_dl, test_files, device, out_dir):
    """Score a trained model on val and test; write metrics.csv, confusion matrix, ROC and predictions."""
    os.makedirs(out_dir, exist_ok=True)
    val_probs, val_labels = predict(model, val_dl, device)
    probs, labels = predict(model, test_dl, device)
    test_ids = [parse(f)[1] for f in test_files]
    rows = [metrics_row("val", name, "clip", val_probs, val_labels),
            metrics_row("test", name, "clip", probs, labels),
            metrics_row("test", name, "participant", *per_participant(probs, labels, test_ids))]
    print("\nTEST (threshold 0.5)")
    for r in rows[1:]:
        print(f"  {r['level']:<12} n={r['n']:<5} balanced acc {r['balanced_acc']:.3f}  "
              f"sensitivity {r['sensitivity']:.3f}  specificity {r['specificity']:.3f}  AUC {r['auc']:.3f}")

    names = sorted(CLASSES, key=CLASSES.get)
    cm = confusion_matrix(labels, (probs >= 0.5).astype(int), labels=[0, 1])
    plot_confusion(cm, names, os.path.join(out_dir, "confusion_matrix_test.png"),
                   title=f"Test set (clips): sensitivity {rows[1]['sensitivity']:.3f}, "
                         f"specificity {rows[1]['specificity']:.3f}")
    fpr, tpr, _ = roc_curve(labels, probs)
    plot_roc(fpr, tpr, roc_auc_score(labels, probs), os.path.join(out_dir, "roc_test.png"),
             title=f"ROC, test set ({name})")
    np.savez(os.path.join(out_dir, "test_predictions.npz"), probs=probs, labels=labels,
             test_files=np.array(test_files))
    save_metrics_csv(rows, os.path.join(out_dir, "metrics.csv"))
    return rows


def main(args):
    grid = load_grid(args.config)
    data_cfg = load_data_config(args.data_config)
    modality = data_cfg["modality"]
    dataset = args.dataset or data_cfg.get("dataset", "../../dataset_build/gait_healthy_and_stroke")
    emg_norm = data_cfg["emg_norm"] if modality in ("emg", "both") else None
    out_dir = args.out_dir or os.path.join("results", modality + ("_per_clip" if emg_norm == "per_clip" else ""))
    model_name = "CA_Transformer" if modality == "both" else "Transformer"
    if modality in ("emg", "both"):
        check_emg_build(dataset)

    print(f"Modality: {modality} ({model_name}"
          f"{f', EMG normalisation {emg_norm}' if emg_norm else ''}); dataset {dataset}")
    train_files, val_files, test_files = make_splits(dataset, modality, args.test_size, args.val_size, args.seed)

    # Normalisation stats from training participants only
    stats = train_stats(train_files, modality)          # {stream: (mean, std)}
    # (with per-clip EMG scaling the EMG stats are stored but not used)
    train_dl = DataLoader(ClipDataset(train_files, stats, modality, emg_norm), batch_size=32, shuffle=True)
    val_dl = DataLoader(ClipDataset(val_files, stats, modality, emg_norm), batch_size=64)
    test_dl = DataLoader(ClipDataset(test_files, stats, modality, emg_norm), batch_size=64)

    device = get_device()
    print(f"Training on {device}; {len(grid)} hyperparameter combination(s) from {args.config}")

    # Grid search: train on train, score on validation
    os.makedirs(out_dir, exist_ok=True)
    results, paths = [], {}
    for i, hp in enumerate(grid):
        name = combo_name(hp)
        print(f"\n=== Combination {i + 1}/{len(grid)}: {name}")
        torch.manual_seed(args.seed)                                    # same init order for every combo
        path, hist = training_loop(build_model(hp, modality), train_dl, labels_for(train_files), val_dl,
                                   epochs=args.epochs, lr=hp["lr"],
                                   out_dir=os.path.join(out_dir, "grid", name),
                                   extra={"hparams": hp, "modality": modality, "emg_norm": emg_norm,
                                          "model": model_name,
                                          "stats": {s: (torch.from_numpy(m), torch.from_numpy(sd))
                                                    for s, (m, sd) in stats.items()}})
        model = build_model(hp, modality).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model_state"])
        row = metrics_row("val", name, "clip", *predict(model, val_dl, device))
        results.append({**hp, "best_epoch": hist["best_epoch"] + 1,
                        **{f"val_{k}": row[k] for k in ["balanced_acc", "auc", "sensitivity", "specificity"]}})
        paths[name] = path
        write_grid_csv(results, os.path.join(out_dir, "grid_search.csv"))  # rewritten every combo

    best = max(results, key=lambda r: (r["val_balanced_acc"], r["val_auc"]))
    best_hp = {k: best[k] for k in GRID_KEYS}
    best_name = combo_name(best_hp)
    print(f"\nBest combination: {best_name}  (val balanced acc {best['val_balanced_acc']:.3f}, "
          f"AUC {best['val_auc']:.3f})")
    with open(os.path.join(out_dir, "best_hparams.json"), "w") as fh:
        json.dump(best_hp, fh, indent=2)
    best_dir = os.path.join(out_dir, "grid", best_name)
    for f in ["best_model.pt", "loss.png", "accuracy.png"]:              # best run's files at the top level
        shutil.copy(os.path.join(best_dir, f), os.path.join(out_dir, f))

    # The chosen model, scored once on test
    model = build_model(best_hp, modality).to(device)
    model.load_state_dict(torch.load(paths[best_name], map_location=device, weights_only=True)["model_state"])
    evaluate_and_save(model, best_name, val_dl, test_dl, test_files, device, out_dir)
    print(f"Plots, predictions, metrics.csv and grid_search.csv saved to {out_dir}/")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_config", type=str, default="data.json",
                        help="Data options: modality (motion | emg | both), emg_norm and dataset path")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset build (overrides 'dataset' in the data config)")
    parser.add_argument("--config", type=str, default="hparams.json",
                        help="Hyperparameter grid (lr, dropout, n_layers, n_heads)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of participants in the fixed test set")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Fraction of the remaining participants used for validation")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None, help="Default: results/<modality>")
    main(parser.parse_args())
