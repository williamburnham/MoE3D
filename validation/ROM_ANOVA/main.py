"""
ROM ANOVA — do joint angle ranges differ across the MoE3D sub-datasets?

Angles come from the AMASS release (`data/MoE3D_data/motion/SMPLH_AMASS`), not from the 263-dim
vectors. In SMPL the rotation stored at a joint is that joint's own rotation
relative to its parent — pose[1] is the thigh w.r.t. the pelvis — so the Euler
components are anatomical by construction.

HumanML3D's `rot_data` was tried first and rejected: its local rotations are
relative to the parent's chained qbetween frame rather than an anatomical joint
coordinate system. The knee happened to agree with the literature (62-72 deg vs
~60) because its parent segment points downward, but the hip came out at 75-111
deg against a literature 40-45, and produced a spurious 28 deg left/right
asymmetry in a healthy subject. It also carries no twist, since qbetween gives
the minimal rotation.
"""

import os
import os.path as osp
import sys
import csv
import re
from glob import glob

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.transform import Rotation

FPS = 20
_RELEASE = osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), '..', '..'))
AMASS_DIR = osp.join(_RELEASE, 'data', 'MoE3D_data', 'motion', 'SMPLH_AMASS')
TRIALS_CSV = osp.join(_RELEASE, 'data', 'MoE3D_data', 'trials.csv')

# SMPL-H pose layout: 156 = 3 root + 21*3 body + 15*3 left hand + 15*3 right hand
BODY0, N_BODY = 3, 21

JOINT_NAMES = [
    'pelvis', 'L_hip', 'R_hip', 'spine1', 'L_knee', 'R_knee', 'spine2',
    'L_ankle', 'R_ankle', 'spine3', 'L_foot', 'R_foot', 'neck', 'L_collar',
    'R_collar', 'head', 'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow',
    'L_wrist', 'R_wrist',
]

# In SMPL the index is the joint itself — no off-by-one. pose[1] rotates the
# thigh, pose[4] the shank, pose[7] the foot.
ANGLE_AT = {
    'L_hip': 1,  'R_hip': 2,
    'L_knee': 4, 'R_knee': 5,
    'L_ankle': 7, 'R_ankle': 8,
    'lumbar': 3, 'thoracic': 6, 'upper_thoracic': 9,
    'neck': 12,
    'L_shoulder': 16, 'R_shoulder': 17,
    'L_elbow': 18,    'R_elbow': 19,
}

COMPONENTS = ('flexion', 'abduction', 'rotation')

GROUPS = ['138H', '50ST', '24MP', '1MPS', 'G120', '9WRT', '10KH', '10KS']

# Normal walking only. G120 also holds stairs, slopes and sit-to-stand; 9WRT
# holds running, jumping, squats and lunges; MovePort holds deliberately
# altered gaits (dragging, leg-high) and standing. Comparing joint ROM across
# sites is meaningless if the groups are doing different things.
# Treadmill vs overground does not matter here: these are joint angles, so belt
# motion does not enter.
# After the rename the action is a field, so gait is simply the two gait codes.
GAIT_ACTIONS = {'GaitGr', 'GaitTm'}


# MovePort imposes three treadmill speeds. Pooling them produces a cohort mean
# that reflects how many trials the protocol ran at each speed rather than how
# the cohort walks: within 24MP alone, low -> high moves ROM by 10-11 deg, about
# twice the size of the between-cohort differences being tested. Each speed is
# therefore its own level.
# ground_gait is kept separate too — per MovePort's readme those trials exist
# only for the participant with spastic paraplegia (Subj25).
# The rename merged MovePort's three belt speeds into one action (GaitTm), so
# the speed is no longer readable from the filename. It survives only in the
# repetition number (high 1-2, low 3-4, medium 5-6 per participant) and in
# trials.csv (imposed_speed). Rebuild the split from there if the speed levels are wanted.
SPEED_SPLIT = {}


# MovePort's three imposed treadmill speeds. The rename merged them into one
# action (GaitTm), so the speed is no longer in the filename -- it survives only
# in trials.csv (imposed_speed, from the original trial stems). The map below is
# exact (163/163 trials, no conflicts) and reproduces MovePort's published
# 63 low / 51 medium / 49 high split. Do NOT infer speed from the repetition
# number: reps 1-2 are reliably 'high' but 3-14 mix low and medium.
#
# Only 'high' is kept. MovePort's readme describes it as 3-4 km/h "comfortable
# brisk walking", the closest of the three to the self-selected overground gait
# every other cohort walked, so it is the only level comparable to them.
# Pooling all three instead makes 24MP's cohort mean a function of how many
# trials the protocol ran at each speed: within 24MP, low -> high moves ROM by
# a mean of 6.7 deg and up to 21.6 deg (L_ankle_flexion_ROM), several times the
# between-cohort differences under test.
KEEP_SPEED = {'24MP': {'high'}}


def _load_speed_map():
    """-> {trial_stem: speed} for cohorts with imposed speeds (trials.csv)."""
    out = {}
    with open(TRIALS_CSV) as fh:
        for r in csv.DictReader(fh):
            if r['subdataset'] in KEEP_SPEED and r['imposed_speed']:
                out[r['trial']] = r['imposed_speed']
    return out


SPEED_MAP = _load_speed_map()


def speed_ok(group, stem):
    """False for a trial at an imposed speed we are not analysing."""
    keep = KEEP_SPEED.get(group)
    if not keep:
        return True
    sp = SPEED_MAP.get(stem)
    if sp is None:
        return True          # not an imposed-speed trial (e.g. overground)
    return sp in keep


def group_label(group, stem):
    """Analysis group for a trial, splitting imposed-speed conditions apart."""
    for pat, lab in SPEED_SPLIT.get(group, ()):
        if pat in stem:
            return lab
    return group


def subject_of(stem):
    """'50ST-Strk-GaitGr-0008-0300' -> '50ST-0008'."""
    f = stem.split('-')
    return f'{f[0]}-{f[3]}'


def class_of(stem):
    """Class is the second field of the name."""
    return {'Hlth': 'Healthy', 'Strk': 'Stroke',
            'SpPa': 'Spastic Paraplegia'}.get(stem.split('-')[1], 'Unknown')


def is_gait(group, stem):
    f = stem.split('-')
    return (len(f) >= 3 and f[2] in GAIT_ACTIONS
            and speed_ok(group, stem))

# Upper-body angles for these groups come from MoSh's pose prior, not markers.
UNMEASURED = {
    '9WRT': {'neck', 'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow'},
    '10KH': {'lumbar', 'thoracic', 'upper_thoracic', 'neck',
             'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow'},
    '10KS': {'lumbar', 'thoracic', 'upper_thoracic', 'neck',
             'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow'},
}



def rest_bone_axes():
    """Dominant rest-pose axis of the bone each joint rotates: {joint: 'X'|'Y'|'Z'}.

    The Euler sequence puts the bone's own long axis last, leaving twist as the
    final component. Computed from the neutral SMPL-H model's rest pose
    (J_regressor @ v_template, child joint minus joint); fixed here because the
    SMPL-H model is licensed separately and not distributed with MoE3D.
    """
    return {'L_hip': 'Y', 'R_hip': 'Y', 'L_knee': 'Y', 'R_knee': 'Y',
            'L_ankle': 'Z', 'R_ankle': 'Z', 'lumbar': 'Y', 'thoracic': 'Y',
            'upper_thoracic': 'Y', 'neck': 'Y', 'L_shoulder': 'X',
            'R_shoulder': 'X', 'L_elbow': 'X', 'R_elbow': 'X'}


# Sequence puts the bone axis last, so component 3 is twist about the bone.
SEQ = {'X': 'ZYX', 'Y': 'XZY', 'Z': 'XYZ'}


def joint_angles(poses):
    """{joint: [T, 3] degrees} as (flexion, abduction, rotation).

    `poses` is the AMASS [T, 156] fullpose array.
    """
    T = len(poses)
    body = poses[:, BODY0:BODY0 + N_BODY * 3].reshape(T, N_BODY, 3)
    axes = rest_bone_axes()
    out = {}
    for name, j in ANGLE_AT.items():
        aa = body[:, j - 1]                       # body block starts at joint 1
        eul = Rotation.from_rotvec(aa).as_euler(SEQ[axes[name]], degrees=True)
        out[name] = eul
    return out


def calculate_min_max_rom(poses, robust=True):
    """
        takes a sample, calculates the minimum, maximum, and ROM of every joint

    One triple per joint per component (flexion / abduction / rotation).
    `robust=True` uses the 5th-95th percentile rather than the raw extremes: a
    single glitched frame otherwise sets both ends and ROM measures the worst
    artefact in the clip instead of the movement.

    Note min and max are relative to the SMPL rest pose, not to clinical
    neutral standing. ROM is unaffected — a constant offset cancels in max-min —
    but the min/max columns are only comparable between clips, not to published
    normative angles.
    """
    row = {}
    for name, eul in joint_angles(poses).items():
        for k, comp in enumerate(COMPONENTS):
            a = eul[:, k]
            if robust:
                lo, hi = np.percentile(a, 5), np.percentile(a, 95)
            else:
                lo, hi = a.min(), a.max()
            row[f'{name}_{comp}_min'] = float(lo)
            row[f'{name}_{comp}_max'] = float(hi)
            row[f'{name}_{comp}_ROM'] = float(hi - lo)
    return row


def calculate_dataset(group, gait_only=True, out_dir='.', robust=True):
    """
        runs calculate_min_max_rom() for every sample in a dataset and saves them all to a csv
    """
    # Recorded trials only: the treadmill 'translated' copies are the same
    # trials re-expressed, and joint angles are identical in both frames.
    files = sorted(f for f in glob(osp.join(AMASS_DIR, '**', group + '-*.npz'),
                                   recursive=True)
                   if osp.sep + 'translated' + osp.sep not in f)
    rows, skipped = [], 0
    for f in files:
        stem = osp.splitext(osp.basename(f))[0]
        if gait_only and not is_gait(group, stem):
            skipped += 1
            continue
        d = np.load(f, allow_pickle=True)
        poses = d['poses']
        row = calculate_min_max_rom(poses, robust=robust)
        # Blank the joints this group never measured, so they cannot contribute
        # a difference that is really MoSh's prior talking.
        for j in UNMEASURED.get(group, ()):
            for c in COMPONENTS:
                for k in ('min', 'max', 'ROM'):
                    row[f'{j}_{c}_{k}'] = np.nan
        row.update(trial=stem, group=group_label(group, stem),
                   source_group=group, subject=subject_of(stem),
                   clinical_class=class_of(stem), n_frames=len(poses),
                   framerate=float(d['mocap_framerate']))
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    path = osp.join(out_dir, f'rom_{group}.csv')
    df.to_csv(path, index=False)
    bits = ''
    if group in SPEED_SPLIT and len(df):
        bits = '  [' + ', '.join(f'{k}:{v}' for k, v in
                                 df.group.value_counts().sort_index().items()) + ']'
    print(f'  {group:<6} {len(df):>5} trials  ({skipped} non-gait skipped)'
          f'  {df["subject"].nunique():>4} subjects{bits}')
    return df


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (scipy 1.7 has no implementation)."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    q[ok] = adj
    return q


def welch_anova(samples):
    """Welch's one-way ANOVA: F, df1, df2, p.

    Each group is weighted by w = n / s^2, i.e. by precision rather than by n
    alone, and the error df is adjusted (Welch-Satterthwaite). Use this when
    group variances differ — Levene rejects for most features here, and the
    group SDs span roughly 5x.

    Caveat worth remembering: precision weighting means a small, wide group
    contributes very little. The clinical groups are exactly the small, wide
    ones, so Welch quietly down-weights them relative to G120 and 138H.
    """
    k = len(samples)
    n = np.array([len(s) for s in samples], float)
    mean = np.array([s.mean() for s in samples])
    var = np.array([s.var(ddof=1) for s in samples])
    if np.any(var <= 0):
        return np.nan, np.nan, np.nan, np.nan
    w = n / var
    sw = w.sum()
    mw = (w * mean).sum() / sw
    A = (w * (mean - mw) ** 2).sum() / (k - 1)
    lam = (((1 - w / sw) ** 2) / (n - 1)).sum()
    B = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam
    F = A / B
    df1 = k - 1
    df2 = (k ** 2 - 1) / (3 * lam)
    return F, df1, df2, float(stats.f.sf(F, df1, df2))


def run_anova(df, groups=None, by='subject', label=''):
    """One-way ANOVA across `groups` for every feature.

    `by='subject'` averages each subject's trials first. Trials from one person
    are not independent: measured ICC is 0.92 for G120 and 0.93 for 50ST, so
    600 G120 trials carry about as much information as 128 observations. A
    trial-level test inflates F by ~3x and is reported only for comparison.
    """
    data = df if groups is None else df[df.group.isin(groups)]
    # The one spastic-paraplegia participant is not a healthy control.
    if groups is not None and label.startswith('healthy'):
        data = data[data.clinical_class != 'Spastic Paraplegia']
    feats = [c for c in df.columns if c.endswith(('_min', '_max', '_ROM'))]
    if by == 'subject':
        data = data.groupby(['group', 'subject'])[feats].mean().reset_index()
    out = []
    for f in feats:
        samples = [g[f].dropna().values for _, g in data.groupby('group')]
        samples = [s for s in samples if len(s) >= 2]
        if len(samples) < 2:
            continue
        F, p = stats.f_oneway(*samples)
        Fw, df1, df2, pw = welch_anova(samples)
        W, pl = stats.levene(*samples, center='median')
        allv = np.concatenate(samples)
        gm = allv.mean()
        ssb = sum(len(s) * (s.mean() - gm) ** 2 for s in samples)
        sst = ((allv - gm) ** 2).sum()
        out.append(dict(contrast=label, feature=f, n_groups=len(samples),
                        n_obs=len(allv), F=F, p=p,
                        welch_F=Fw, welch_df2=df2, welch_p=pw,
                        eta_sq=ssb / sst if sst > 0 else np.nan,
                        levene_W=W, levene_p=pl))
    res = pd.DataFrame(out)
    if len(res):
        res['p_fdr'] = bh_fdr(res['p'])
        res['welch_p_fdr'] = bh_fdr(res['welch_p'])
    return res.sort_values('welch_p')


HEALTHY_GROUPS = ['138H', '24MP', 'G120', '9WRT', '10KH']
STROKE_GROUPS = ['50ST', '10KS']




def drop_outliers_iqr(df, k=1.5, verbose=True):
    """Blank values beyond k x IQR of their own group, per feature.

    Two choices that matter:

    * Bounds are computed **within each group**, not across the corpus. A stroke
      participant's 17 deg knee ROM is ordinary for that cohort but extreme
      against the whole dataset; a global rule would delete the clinical signal
      this analysis exists to demonstrate.
    * Individual cells are blanked, not whole trials. A trial whose ankle fit
      failed usually still has a usable knee, and dropping the row would throw
      that away.

    Note this necessarily shrinks within-group variance, so F statistics rise
    and Levene is affected. The unfiltered results stay available for comparison
    — see `--no-filter`.
    """
    feats = [c for c in df.columns if c.endswith(('_min', '_max', '_ROM'))]
    out = df.copy()
    removed = {}
    for g, idx in df.groupby('group').groups.items():
        block = out.loc[idx, feats]
        q1 = block.quantile(0.25)
        q3 = block.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask = block.lt(lo, axis=1) | block.gt(hi, axis=1)
        out.loc[idx, feats] = block.mask(mask)
        removed[g] = int(mask.values.sum())
    if verbose:
        tot = sum(removed.values())
        cells = len(df) * len(feats)
        print(f'  outlier filter ({k}x IQR within group): blanked {tot} of '
              f'{cells} cells ({100 * tot / cells:.2f}%)')
        for g, n in sorted(removed.items(), key=lambda x: -x[1])[:5]:
            gn = (df.group == g).sum() * len(feats)
            print(f'     {g:<13} {n:>5} of {gn:>6} ({100 * n / gn:.1f}%)')
    return out


def participant_table(df, out_dir='.'):
    """One row per participant: every ROM feature, averaged over their trials.

    Columns are ordered so the trustworthy measures come first — lower-limb
    flexion validates against published gait norms, whereas the abduction and
    rotation components, and the spine and arm joints, are exploratory.
    Blank cells are joints the group never measured (NaN by design, not error).
    """
    rom = [c for c in df.columns if c.endswith('_ROM')]
    limb = ['L_hip', 'R_hip', 'L_knee', 'R_knee', 'L_ankle', 'R_ankle']
    first = [f'{j}_flexion_ROM' for j in limb]
    rest = [c for c in rom if c not in first]
    order = first + sorted(rest)

    g = df.groupby(['group', 'subject'])
    out = g[order].mean().round(2)
    out.insert(0, 'n_trials', g.size())
    out.insert(1, 'clinical_class', g['clinical_class'].first())
    out.insert(2, 'mean_trial_frames', g['n_frames'].mean().round(0).astype(int))
    out = out.reset_index().sort_values(['group', 'subject'])
    path = osp.join(out_dir, 'participant_rom.csv')
    out.to_csv(path, index=False)
    print(f'wrote {osp.basename(path)}: {len(out)} participants x {len(order)} ROM features')
    return out



PAIRWISE_FEATURES = ['L_hip_flexion_ROM', 'R_hip_flexion_ROM',
                     'L_knee_flexion_ROM', 'R_knee_flexion_ROM']


def tukey_kramer(df, features=PAIRWISE_FEATURES, out_dir='.', alpha=0.05):
    """Tukey-Kramer post-hoc across every pair of groups, per feature.

    The Kramer extension of Tukey's HSD, for unequal group sizes. The
    studentized range distribution controls the family-wise error rate across
    all pairs within a feature, so no further correction is applied — that is
    the point of the test, unlike a set of t-tests which would need one.

    Subject means are the unit: trials within a participant are not independent
    (ICC up to 0.93), so a trial-level test would inflate the statistic roughly
    threefold.

    Assumption to keep in view: Tukey-Kramer pools one within-group variance
    across all groups, so it assumes homogeneity of variance. Levene rejects
    that for most features here (group SDs span roughly 5x), which makes the
    pooled MSE a compromise between wide and narrow groups.
    """
    feats = [f for f in features if f in df.columns]
    sub = df.groupby(['group', 'subject'])[feats].mean().reset_index()
    rows = []
    for f in feats:
        d = sub[['group', f]].dropna()
        groups = [(g, v[f].values) for g, v in d.groupby('group')
                  if len(v) >= 2]
        if len(groups) < 2:
            continue
        k = len(groups)
        N = sum(len(v) for _, v in groups)
        dof = N - k
        # Pooled within-group mean square — the shared error term.
        mse = sum(((v - v.mean()) ** 2).sum() for _, v in groups) / dof
        qcrit = stats.studentized_range.ppf(1 - alpha, k, dof)
        for i, (a_, va) in enumerate(groups):
            for b_, vb in groups[i + 1:]:
                diff = va.mean() - vb.mean()
                se = np.sqrt(mse / 2 * (1 / len(va) + 1 / len(vb)))
                q = abs(diff) / se
                p = float(stats.studentized_range.sf(q, k, dof))
                rows.append(dict(
                    feature=f, group_a=a_, group_b=b_,
                    n_a=len(va), n_b=len(vb),
                    mean_a=va.mean(), mean_b=vb.mean(), mean_diff=diff,
                    se=se, q=q, p_tukey=p, k=k, dof=dof,
                    ci_lo=diff - qcrit * se, ci_hi=diff + qcrit * se,
                    reject=p < alpha))
    res = pd.DataFrame(rows).sort_values(['feature', 'p_tukey'])
    path = osp.join(out_dir, 'tukey_kramer.csv')
    res.to_csv(path, index=False)
    npair = len(res) // max(len(feats), 1)
    print(f'wrote {osp.basename(path)}: {len(res)} comparisons '
          f'({len(feats)} features x {npair} pairs), FWER controlled within feature')
    return res


def main():
    """
        - runs calculate_dataset() on each dataset
        - runs anova between all results
    """
    here = osp.dirname(osp.abspath(__file__))
    print('per-trial ROM, normal gait only:')
    frames = [calculate_dataset(g, out_dir=here) for g in GROUPS]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(osp.join(here, 'rom_all_unfiltered.csv'), index=False)
    if '--no-filter' not in sys.argv:
        df = drop_outliers_iqr(df)
    df.to_csv(osp.join(here, 'rom_all.csv'), index=False)
    print(f'\ntotal {len(df)} trials, {df["subject"].nunique()} subjects, '
          f'{df["group"].nunique()} groups\n')

    participant_table(df, out_dir=here)
    tukey_kramer(df, out_dir=here)

    contrasts = [
        ('healthy', HEALTHY_GROUPS),
        ('stroke', STROKE_GROUPS),
        ('all', None),
    ]
    parts = []
    for label, groups in contrasts:
        res = run_anova(df, groups=groups, label=label)
        parts.append(res)
        sig = int((res.welch_p_fdr < 0.05).sum()) if len(res) else 0
        sub = df if groups is None else df[df.group.isin(groups)]
        if groups is not None and label == 'healthy':
            sub = sub[sub.clinical_class != 'Spastic Paraplegia']
        ng, nsub = sub.group.nunique(), sub.subject.nunique()
        # Upper-body features cover fewer groups: the masked joints are NaN for
        # 9WRT and Knee-Flex, so n_groups varies by feature.
        rng = f'{res.n_groups.min()}-{res.n_groups.max()}' if len(res) else '-'
        note = '  (two-sample test)' if ng == 2 else ''
        print(f'  {label:<8} {ng} groups, {nsub:>4} subjects  '
              f'(groups per feature {rng})  '
              f'{sig:>3}/{len(res)} significant at FDR 0.05{note}')
    allres = pd.concat(parts, ignore_index=True)
    allres.to_csv(osp.join(here, 'anova_contrasts.csv'), index=False)

    trial = run_anova(df, by='trial', label='all (trial-level)')
    trial.to_csv(osp.join(here, 'anova_trial_level.csv'), index=False)
    print('\nwrote anova_contrasts.csv and anova_trial_level.csv')
    return df, allres


if __name__ == '__main__':
    main()
