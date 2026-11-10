"""Score the N-th ranked combination of a finished grid search on the test set.

    python evaluate.py --results results/both_per_clip --rank 3

Reads <results>/grid_search.csv, loads that combination's best_model.pt (modality, EMG
normalisation, hyperparameters and training-set stats are stored in the checkpoint),
rebuilds the same participant split, and writes the same outputs as main.py
(metrics.csv, confusion matrix, ROC, predictions, loss/accuracy curves) to <results>/<rank>,
e.g. results/both_per_clip/3rd. --test_size, --val_size and --seed must match the training run.
"""
import os
import csv
import shutil
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataset import ClipDataset, train_stats, load_data_config
from model.train_model import get_device
from main import GRID_KEYS, combo_name, build_model, make_splits, evaluate_and_save


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def main(args):
    with open(os.path.join(args.results, "grid_search.csv"), newline="") as fh:
        ranked = list(csv.DictReader(fh))                 # already sorted best first
    if not 1 <= args.rank <= len(ranked):
        raise ValueError(f"--rank must be 1..{len(ranked)}")
    row = ranked[args.rank - 1]
    hp = {"lr": float(row["lr"]), "dropout": float(row["dropout"]),
          "n_layers": int(row["n_layers"]), "n_heads": int(row["n_heads"])}
    name = combo_name(hp)
    ckpt_path = os.path.join(args.results, "grid", name, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    modality, emg_norm = ckpt["modality"], ckpt["emg_norm"]
    assert {k: ckpt["hparams"][k] for k in GRID_KEYS} == hp, "checkpoint does not match grid_search.csv"
    print(f"Rank {args.rank}: {name} ({modality}, EMG normalisation {emg_norm}), "
          f"saved epoch {ckpt['epoch'] + 1}, val balanced acc {row['val_balanced_acc']}")

    dataset = args.dataset or load_data_config(args.data_config).get("dataset")
    train_files, val_files, test_files = make_splits(dataset, modality, args.test_size, args.val_size, args.seed)

    # Check the rebuilt split is the training run's: its training-set stats must match the checkpoint's
    stats = {s: (m.numpy(), sd.numpy()) for s, (m, sd) in ckpt["stats"].items()}
    for s, (m, sd) in train_stats(train_files, modality).items():
        if not (np.allclose(m, stats[s][0], atol=1e-5) and np.allclose(sd, stats[s][1], atol=1e-5)):
            raise RuntimeError(f"{s} stats differ from the checkpoint: the split does not match the "
                               "training run (check --dataset, --test_size, --val_size, --seed)")
    print("Split check: training-set stats match the checkpoint")

    device = get_device()
    model = build_model(hp, modality).to(device)
    model.load_state_dict(ckpt["model_state"])
    val_dl = DataLoader(ClipDataset(val_files, stats, modality, emg_norm), batch_size=64)
    test_dl = DataLoader(ClipDataset(test_files, stats, modality, emg_norm), batch_size=64)

    out_dir = os.path.join(args.results, ordinal(args.rank))
    rows = evaluate_and_save(model, name, val_dl, test_dl, test_files, device, out_dir)
    if abs(rows[0]["balanced_acc"] - float(row["val_balanced_acc"])) > 1e-3:
        print(f"WARNING: val balanced acc {rows[0]['balanced_acc']:.4f} differs from grid_search.csv")
    for f in ["loss.png", "accuracy.png", "best_model.pt"]:
        shutil.copy(os.path.join(args.results, "grid", name, f), os.path.join(out_dir, f))
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="A main.py output folder, e.g. results/both")
    parser.add_argument("--rank", type=int, default=3, help="Rank in grid_search.csv (1 = best)")
    parser.add_argument("--data_config", type=str, default="data.json", help="Only its 'dataset' path is used")
    parser.add_argument("--dataset", type=str, default=None, help="Overrides the data config's dataset path")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
