#!/usr/bin/env python3
"""
build_dataset.py — assemble one version of MoE3D into dataset_build/<name>/.

Choose the motion representation, the EMG type, which trials to include and
which sub-datasets, and the script copies exactly that selection into a new
folder, cuts the matching EMG for every motion file, and computes Mean/Std over
the selection. See README.md for a tutorial.

    python build_dataset.py --motion 263dim_HumanML3D_normalisation \\
                            --emg processed --trials all_translated
    python build_dataset.py --config config_example.json

Output (dataset_build/<name>/):
    motion/new_joint_vecs/*.npy  motion/new_joints/*.npy    (263-dim bases)
    motion/*.npz                                             (SMPLH_AMASS)
    emg/*.npy  (processed)   or   emg/*.npz  (raw: emg, fs, labels)
    Mean.npy, Std.npy            (263-dim bases)
    emg_Mean.npy, emg_Std.npy    (when EMG is included)
    build_info.json              (options, config path, counts, unpaired files)

Requires numpy only.
"""

import argparse
import csv
import datetime
import json
import os
import os.path as osp
import shutil
import sys

import numpy as np

ROOT = osp.dirname(osp.abspath(__file__))
DATA = osp.join(ROOT, 'data', 'MoE3D_data')
MOTION = osp.join(DATA, 'motion')
EMG = osp.join(DATA, 'EMG')
OUT = osp.join(ROOT, 'dataset_build')

MOTION_BASES = ['263dim_original_proportions', '263dim_HumanML3D_normalisation',
                'SMPLH_AMASS']
EMG_TYPES = ['processed', 'raw', 'none']
SUBDATASETS = ['138H', '50ST', '24MP', '1MPS', 'G120', '9WRT', '10KH', '10KS']

# Trial-type option -> [(category, version)]. Overground categories have no
# version; treadmill categories are either 'original' (recorded on the belt)
# or 'translated' (re-expressed as overground locomotion).
TRIALS = {
    'overground_gait_only': [('gait_overground', None)],
    'original_treadmill_gait_only': [('gait_treadmill', 'original')],
    'translated_treadmill_gait_only': [('gait_treadmill', 'translated')],
    'overground_and_original_treadmill_gait': [('gait_overground', None),
                                               ('gait_treadmill', 'original')],
    'overground_and_translated_treadmill_gait': [('gait_overground', None),
                                                 ('gait_treadmill', 'translated')],
    'all_original': [('gait_overground', None), ('gait_treadmill', 'original'),
                     ('other_overground', None), ('other_treadmill', 'original')],
    'all_translated': [('gait_overground', None), ('gait_treadmill', 'translated'),
                       ('other_overground', None), ('other_treadmill', 'translated')],
}

PROCESSED_RATE = 100.0      # Hz, processed EMG envelopes
JOINTS_NUM = 22

# Joints with no marker behind them, zeroed in the 263-dim files. Excluded from
# Mean/Std so MoSh's pose prior does not set the scaling for groups that did
# measure them.
SPINE, NECK_HEAD, ARMS = (3, 6, 9), (12, 13, 14, 15), (16, 17, 18, 19, 20, 21)
UNMEASURED = {'9WRT': NECK_HEAD + ARMS,
              '10KH': SPINE + NECK_HEAD + ARMS,
              '10KS': SPINE + NECK_HEAD + ARMS}


# ── names ────────────────────────────────────────────────────────────────────

def parse(name):
    """'M24MP-Hlth-GaitTm-0013-0500.npy' -> (stem, subdataset, mirrored)."""
    stem = osp.splitext(osp.basename(name))[0]
    mirrored = stem.startswith('M')
    return stem, (stem[1:] if mirrored else stem).split('-')[0], mirrored


def unmirrored(stem):
    return stem[1:] if stem.startswith('M') else stem


def read_csv(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


# ── selection ────────────────────────────────────────────────────────────────

def category_dir(base_dir, cat, version):
    return osp.join(base_dir, cat, version) if version else osp.join(base_dir, cat)


def motion_files(motion, trials, subdatasets):
    """[(path, category, version)] for the chosen selection, sorted by name."""
    if motion == 'SMPLH_AMASS':
        root = osp.join(MOTION, motion)
    else:
        root = osp.join(MOTION, motion, 'new_joint_vecs')
    out = []
    for cat, version in TRIALS[trials]:
        d = category_dir(root, cat, version)
        if not osp.isdir(d):
            continue
        for f in os.listdir(d):
            if parse(f)[1] in subdatasets:
                out.append((osp.join(d, f), cat, version))
    return sorted(out, key=lambda x: osp.basename(x[0]))


# ── EMG cutting ──────────────────────────────────────────────────────────────

class EMGSource:
    """Per-trial EMG plus the indices that place each motion file inside it."""

    def __init__(self, kind):
        self.kind = kind
        self.ext = '.npy' if kind == 'processed' else '.npz'
        self.path = {}
        for dirpath, _, files in os.walk(osp.join(EMG, kind)):
            for f in files:
                if f.endswith(self.ext):
                    self.path[osp.splitext(f)[0]] = osp.join(dirpath, f)
        with open(osp.join(EMG, 'emg_channels.json')) as fh:
            self.labels = json.load(fh)['labels']
        pos = {l: i for i, l in enumerate(self.labels)}
        swap = {'R': 'L', 'L': 'R'}
        self.mirror = [pos.get(swap.get(l[:1], l[:1]) + l[1:], i)
                       for i, l in enumerate(self.labels)]
        self.trials = {r['trial']: r for r in read_csv(osp.join(DATA, 'trials.csv'))}
        self.clip = {}
        for fname, version in (('index_2.csv', 'original'),
                               ('index_2_translated.csv', 'translated')):
            for r in read_csv(osp.join(DATA, fname)):
                self.clip[(osp.splitext(r['new_name'])[0], version)] = (
                    r['trial'], int(r['start_frame']), int(r['end_frame']))

    def load(self, trial):
        if self.kind == 'processed':
            return np.load(self.path[trial]), PROCESSED_RATE
        z = np.load(self.path[trial])
        return z['emg'], float(z['fs'])

    def seconds_per_frame(self, trial):
        # Every 263-dim file is exactly 20 fps (motion_fps in trials.csv);
        # 9WRT, captured at 128 Hz, was interpolated onto the 20 fps grid.
        return 1.0 / float(self.trials[trial]['motion_fps'])

    def for_clip(self, stem, version):
        """EMG for one 263-dim clip, or None when the trial has no EMG."""
        key = (unmirrored(stem), version or 'original')
        if key not in self.clip:
            return None
        trial, s, e = self.clip[key]
        if trial not in self.path:
            return None
        emg, rate = self.load(trial)
        spf = self.seconds_per_frame(trial)
        a = int(round(s * spf * rate))
        n = int(round((e - s) * spf * rate))
        piece = emg[:, a:a + n]
        if piece.shape[1] < n:      # resampling can land a few samples short
            piece = np.concatenate([piece, np.zeros((emg.shape[0], n - piece.shape[1]),
                                                    emg.dtype)], axis=1)
        if stem.startswith('M'):
            piece = piece[self.mirror]
        return piece, rate

    def for_trial(self, stem, version):
        """EMG for one SMPL-H trial; translated trials are trimmed to match."""
        if stem not in self.path:
            return None
        emg, rate = self.load(stem)
        if version == 'translated':
            t = self.trials[stem]
            fps = float(t['mocap_fps'])
            a = int(round(int(t['trim_start_frame']) / fps * rate))
            b = int(round(int(t['trim_end_frame']) / fps * rate))
            emg = emg[:, a:b]
        return emg, rate

    def save(self, path_no_ext, emg, rate):
        if self.kind == 'processed':
            np.save(path_no_ext + '.npy', emg)
        else:
            np.savez(path_no_ext + '.npz', emg=emg, fs=np.float64(rate),
                     labels=np.array(self.labels))


# ── statistics ───────────────────────────────────────────────────────────────

def joint_dims(joints, n=JOINTS_NUM):
    d = []
    for j in joints:
        if j >= 1:
            d += range(4 + (j - 1) * 3, 4 + (j - 1) * 3 + 3)                 # ric
            d += range(4 + (n - 1) * 3 + (j - 1) * 6,
                       4 + (n - 1) * 3 + (j - 1) * 6 + 6)                    # rot
        d += range(4 + (n - 1) * 9 + j * 3, 4 + (n - 1) * 9 + j * 3 + 3)      # vel
    return np.array(sorted(set(d)), dtype=np.int64)


class Moments:
    """Running per-dimension sum / sum of squares / count, with a validity mask."""

    def __init__(self, dim):
        self.s = np.zeros(dim)
        self.q = np.zeros(dim)
        self.n = np.zeros(dim)

    def add(self, x, valid):
        x = x.astype(np.float64)
        self.s[valid] += x[:, valid].sum(0)
        self.q[valid] += (x[:, valid] ** 2).sum(0)
        self.n[valid] += x.shape[0]

    def result(self):
        n = np.maximum(self.n, 1)
        mean = self.s / n
        std = np.sqrt(np.maximum(self.q / n - mean ** 2, 0))
        return mean, std


def motion_stats(m):
    """HumanML3D convention: per-dimension Mean, one Std shared per block."""
    mean, std = m.result()
    n = JOINTS_NUM
    for a, b in [(0, 1), (1, 3), (3, 4), (4, 4 + (n - 1) * 3),
                 (4 + (n - 1) * 3, 4 + (n - 1) * 9),
                 (4 + (n - 1) * 9, 4 + (n - 1) * 9 + n * 3),
                 (4 + (n - 1) * 9 + n * 3, 8 + (n - 1) * 9 + n * 3)]:
        std[a:b] = std[a:b].mean()
    std[std < 1e-8] = 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


# ── build ────────────────────────────────────────────────────────────────────

def resolve_subdatasets(include, exclude):
    chosen = SUBDATASETS if include == ['all'] else include
    bad = [s for s in chosen + exclude if s not in SUBDATASETS]
    if bad:
        raise SystemExit('unknown sub-dataset(s) %s; choose from %s'
                         % (bad, ', '.join(SUBDATASETS)))
    return [s for s in chosen if s not in exclude]


def build(a):
    subdatasets = resolve_subdatasets(a.subdatasets, a.exclude)
    if not subdatasets:
        raise SystemExit('no sub-datasets left after --exclude')
    name = a.name or datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if any(c in name for c in '/\\:*?"<>|'):
        raise SystemExit('--name may not contain / \\ : * ? " < > |')
    out = osp.join(OUT, name)
    if osp.exists(out):
        raise SystemExit('%s already exists; choose another --name' % out)

    files = motion_files(a.motion, a.trials, subdatasets)
    if not files:
        raise SystemExit('the selection is empty: %s / %s / %s'
                         % (a.motion, a.trials, ', '.join(subdatasets)))
    is263 = a.motion != 'SMPLH_AMASS'
    emg = None if a.emg == 'none' else EMGSource(a.emg)

    print('building %s\n  motion %s | emg %s | trials %s\n  sub-datasets %s\n  %d motion files'
          % (out, a.motion, a.emg, a.trials, ', '.join(subdatasets), len(files)))

    mot = osp.join(out, 'motion')
    if is263:
        os.makedirs(osp.join(mot, 'new_joint_vecs'))
        os.makedirs(osp.join(mot, 'new_joints'))
    else:
        os.makedirs(mot)
    if emg:
        os.makedirs(osp.join(out, 'emg'))

    vec_m = Moments(263) if is263 else None
    emg_m = Moments(len(emg.labels)) if emg else None
    masks = {g: joint_dims(j) for g, j in UNMEASURED.items()}
    counts, unpaired = {}, []
    for i, (src, cat, version) in enumerate(files):
        fname = osp.basename(src)
        stem, group, _ = parse(fname)
        if is263:
            shutil.copyfile(src, osp.join(mot, 'new_joint_vecs', fname))
            jsrc = src.replace(osp.sep + 'new_joint_vecs' + osp.sep,
                               osp.sep + 'new_joints' + osp.sep)
            shutil.copyfile(jsrc, osp.join(mot, 'new_joints', fname))
            v = np.load(src)
            valid = np.ones(263, bool)
            if group in masks:
                valid[masks[group]] = False
            vec_m.add(v, valid)
        else:
            shutil.copyfile(src, osp.join(mot, fname))

        if emg:
            got = (emg.for_clip(stem, version) if is263
                   else emg.for_trial(stem, version))
            if got is None:
                unpaired.append(stem)
            else:
                x, rate = got
                emg.save(osp.join(out, 'emg', stem), x, rate)
                # A channel that is all zero in this file was not recorded.
                rec = np.any(x != 0, axis=1)
                emg_m.add(x.T, rec)
        counts[group] = counts.get(group, 0) + 1
        if (i + 1) % 2000 == 0:
            print('  %d/%d' % (i + 1, len(files)), flush=True)

    if is263:
        mean, std = motion_stats(vec_m)
        np.save(osp.join(out, 'Mean.npy'), mean)
        np.save(osp.join(out, 'Std.npy'), std)
    if emg:
        em, es = emg_m.result()
        es[es < 1e-8] = 1.0          # never-recorded channel: leave it unscaled
        np.save(osp.join(out, 'emg_Mean.npy'), em.astype(np.float32))
        np.save(osp.join(out, 'emg_Std.npy'), es.astype(np.float32))

    info = dict(name=name, created=datetime.datetime.now().isoformat(timespec='seconds'),
                motion=a.motion, emg=a.emg, trials=a.trials,
                trial_categories=['%s/%s' % (c, v) if v else c for c, v in TRIALS[a.trials]],
                subdatasets=subdatasets, motion_files=len(files),
                motion_files_per_subdataset=counts,
                emg_files=(len(files) - len(unpaired)) if emg else 0,
                unpaired_motion=sorted(unpaired),
                emg_labels=emg.labels if emg else None,
                emg_rate_hz=(PROCESSED_RATE if a.emg == 'processed'
                             else 'native, per file (fs)' if a.emg == 'raw' else None),
                config=a.config,
                command=' '.join(sys.argv))
    with open(osp.join(out, 'build_info.json'), 'w') as fh:
        json.dump(info, fh, indent=1)
    print('done: %d motion files%s -> %s'
          % (len(files), (', %d with EMG, %d unpaired' % (info['emg_files'], len(unpaired)))
             if emg else '', out))


def list_available():
    trials = read_csv(osp.join(DATA, 'trials.csv'))
    print('sub-dataset  trials  categories')
    for g in SUBDATASETS:
        rows = [t for t in trials if t['subdataset'] == g]
        cats = {}
        for t in rows:
            cats[t['category']] = cats.get(t['category'], 0) + 1
        print('  %-9s %6d  %s' % (g, len(rows), ', '.join('%s %d' % kv for kv in sorted(cats.items()))))
    print('\nmotion bases: %s\nemg types:    %s\ntrial types:\n  %s'
          % (', '.join(MOTION_BASES), ', '.join(EMG_TYPES), '\n  '.join(TRIALS)))


# Options a config file may set, with their defaults. --motion and --trials
# have none: they must come from the command line or the config.
DEFAULTS = dict(motion=None, emg='processed', trials=None,
                subdatasets=['all'], exclude=[], name=None)


def load_config(path):
    """Read a JSON config of build options; keys starting with '_' are comments."""
    try:
        with open(path) as fh:
            cfg = json.load(fh)
    except (OSError, ValueError) as e:
        raise SystemExit('cannot read config %s: %s' % (path, e))
    if not isinstance(cfg, dict):
        raise SystemExit('config %s must hold one JSON object' % path)
    cfg = {k: v for k, v in cfg.items() if not k.startswith('_')}
    unknown = sorted(set(cfg) - set(DEFAULTS))
    if unknown:
        raise SystemExit('config %s: unknown key(s) %s; allowed: %s'
                         % (path, ', '.join(unknown), ', '.join(DEFAULTS)))
    for k, choices in (('motion', MOTION_BASES), ('emg', EMG_TYPES), ('trials', list(TRIALS))):
        if k in cfg and cfg[k] not in choices:
            raise SystemExit('config %s: %s must be one of %s'
                             % (path, k, ', '.join(choices)))
    for k in ('subdatasets', 'exclude'):
        if isinstance(cfg.get(k), str):
            cfg[k] = [cfg[k]]
        if k in cfg and not (isinstance(cfg[k], list)
                             and all(isinstance(x, str) for x in cfg[k])):
            raise SystemExit('config %s: %s must be a list of sub-dataset names' % (path, k))
    if cfg.get('name') is not None and not isinstance(cfg['name'], str):
        raise SystemExit('config %s: name must be a string or null' % path)
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', metavar='JSON',
                   help='read the options below from a JSON file; '
                        'options given on the command line override it')
    p.add_argument('--motion', choices=MOTION_BASES, help='motion representation')
    p.add_argument('--emg', choices=EMG_TYPES,
                   help='EMG to include (default: processed)')
    p.add_argument('--trials', choices=list(TRIALS), help='which trials to include')
    p.add_argument('--subdatasets', nargs='+', metavar='SD',
                   help="'all' (default) or a list, e.g. 138H 50ST G120")
    p.add_argument('--exclude', nargs='+', metavar='SD',
                   help='sub-datasets to leave out, e.g. 1MPS')
    p.add_argument('--name', help='output folder name (default: current date and time)')
    p.add_argument('--list', action='store_true',
                   help='show what is available and exit')
    args = p.parse_args()
    if not osp.isfile(osp.join(DATA, 'trials.csv')):
        raise SystemExit('no data found in %s; download the MoE3D data and put it '
                         'there (see README.md)' % DATA)
    if args.list:
        return list_available()
    opts = dict(DEFAULTS)
    if args.config:
        opts.update(load_config(args.config))
    opts.update({k: getattr(args, k) for k in DEFAULTS if getattr(args, k) is not None})
    if not opts['motion'] or not opts['trials']:
        p.error('motion and trials are required, on the command line or in --config '
                '(or use --list)')
    build(argparse.Namespace(config=args.config and osp.abspath(args.config), **opts))


if __name__ == '__main__':
    main()
