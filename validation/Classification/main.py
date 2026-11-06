import os
import glob
from argparse import ArgumentParser
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from torch.utils.data import DataLoader

from dataset.dataset import CLASSES, MAX_LEN, parse, MotionDataset
from model.transformer import Transformer
from model.train_model import training_loop, predict, balanced_accuracy, get_device
from plots import plot_confusion, plot_roc


def files_for(files, participants):
    participants = set(participants)
    return [f for f in files if parse(f)[1] in participants]


def labels_for(files):
    return np.array([parse(f)[0] for f in files])


def main(args):
    torch.manual_seed(args.seed)
    files = sorted(glob.glob(f"{args.dataset}/motion/new_joint_vecs/*.npy"))
    files = [f for f in files if parse(f)[0] is not None]          # keep Hlth/Strk only

    # All splits are over unique participants (not files), so no participant is in two sets
    prt_label = {parse(f)[1]: parse(f)[0] for f in files}
    participants = np.array(sorted(prt_label))
    prt_labels = np.array([prt_label[p] for p in participants])
    print(f"{len(files)} clips, {len(participants)} participants "
          f"({np.bincount(prt_labels).tolist()} per class)")

    # Fixed test set, locked away until the very end
    prt_cv, prt_test, lab_cv, _ = train_test_split(participants, prt_labels, stratify=prt_labels,
                                                   test_size=args.test_size, random_state=args.seed)
    test_files = files_for(files, prt_test)
    print(f"CV pool: {len(prt_cv)} participants; fixed test: {len(prt_test)} participants "
          f"({len(test_files)} clips)")

    mean, std = np.load(f"{args.dataset}/Mean.npy"), np.load(f"{args.dataset}/Std.npy")
    device = get_device()
    print(f"Training on {device}")

    # k-fold CV on the pool: each fold validates on 1/k of the pool, trains on the rest
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    best_paths, val_rocs, val_bal_acc = [], [], []
    for k, (train_idx, val_idx) in enumerate(skf.split(prt_cv, lab_cv)):
        print(f"\nFold {k + 1}/{args.folds}")
        train_files = files_for(files, prt_cv[train_idx])
        val_files = files_for(files, prt_cv[val_idx])
        print(f"  clips: train {len(train_files)}, val {len(val_files)}")

        train_dl = DataLoader(MotionDataset(train_files, mean, std), batch_size=32, shuffle=True)
        val_dl = DataLoader(MotionDataset(val_files, mean, std), batch_size=64)

        model = Transformer(max_len=MAX_LEN)                           # fresh model per fold
        best_paths.append(training_loop(model, train_dl, labels_for(train_files), val_dl,
                                        epochs=args.epochs, lr=args.lr,
                                        out_dir=os.path.join(args.out_dir, f"fold{k + 1}")))

        model.load_state_dict(torch.load(best_paths[-1], map_location=device)["model_state"])
        probs, labels = predict(model, val_dl, device)
        fpr, tpr, _ = roc_curve(labels, probs)
        val_rocs.append((fpr, tpr, roc_auc_score(labels, probs)))
        val_bal_acc.append(balanced_accuracy(probs, labels))
        print(f"  VAL balanced acc {val_bal_acc[-1]:.3f}  AUC {val_rocs[-1][2]:.3f}")

    val_aucs, val_accs = np.array([a for _, _, a in val_rocs]), np.array(val_bal_acc)
    print(f"\n{args.folds}-fold CV (validation)  balanced acc {val_accs.mean():.3f} +/- {val_accs.std():.3f}  "
          f"AUC {val_aucs.mean():.3f} +/- {val_aucs.std():.3f}")
    plot_roc(val_rocs, os.path.join(args.out_dir, "roc_cv_validation.png"),
             title=f"ROC, {args.folds}-fold CV (validation folds)")

    # Final model = ensemble of the k fold models (mean stroke probability); scored once on test
    test_dl = DataLoader(MotionDataset(test_files, mean, std), batch_size=64)
    member_probs = []
    for path in best_paths:
        model = Transformer(max_len=MAX_LEN).to(device)
        model.load_state_dict(torch.load(path, map_location=device)["model_state"])
        p, labels = predict(model, test_dl, device)
        member_probs.append(p)
    probs = np.mean(member_probs, axis=0)
    fpr, tpr, _ = roc_curve(labels, probs)
    test_auc, test_acc = roc_auc_score(labels, probs), balanced_accuracy(probs, labels)
    print(f"\nTEST (ensemble of {args.folds})  balanced acc {test_acc:.3f}  AUC {test_auc:.3f}")

    names = sorted(CLASSES, key=CLASSES.get)
    cm = confusion_matrix(labels, (probs >= 0.5).astype(int), labels=[0, 1])
    plot_confusion(cm, names, os.path.join(args.out_dir, "confusion_matrix_test.png"),
                   title="Test set confusion matrix (clip level)")
    member_rocs = [(*roc_curve(labels, p)[:2], roc_auc_score(labels, p)) for p in member_probs]
    plot_roc(member_rocs, os.path.join(args.out_dir, "roc_test.png"),
             title="ROC, fixed test set", main=(fpr, tpr, test_auc))
    np.savez(os.path.join(args.out_dir, "test_predictions.npz"), probs=probs, labels=labels,
             member_probs=np.array(member_probs), test_files=np.array(test_files),
             val_bal_acc=val_accs, val_auc=val_aucs)
    print(f"Plots and predictions saved to {args.out_dir}/")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../../dataset_build/gait_healthy_and_stroke",
                        help="Path to dataset")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of participants in the fixed test set")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results")
    main(parser.parse_args())
