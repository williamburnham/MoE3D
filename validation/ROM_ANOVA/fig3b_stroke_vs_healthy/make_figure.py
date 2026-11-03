"""Fig 3b — as fig 3, but keeping the stroke-vs-healthy pairs.

Fig 3 drops mixed pairs deliberately: they restate that the clinical cohorts
differ from the healthy ones, which the class contrast already establishes.
This variant keeps all 21 pairs so the size of the clinical separation can be
read against the between-cohort differences. Exploratory -- the Tukey-Kramer
intervals are the same ones fig 3 draws, computed over all 8 groups.
"""
import os.path as osp, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from _figstyle import (CLASS_COLOUR, INK, MUTED, GRID, TREADMILL_GROUPS,
                       PAIR_TREADMILL, PAIR_MODALITY, PAIR_MIXED, tidy)

HERE = osp.dirname(osp.abspath(__file__))
r = pd.read_csv(osp.join(HERE, '..', 'tukey_kramer.csv'))
CLINICAL = {'50ST', '10KS'}
FEATS = ['L_hip_flexion_ROM', 'R_hip_flexion_ROM',
         'L_knee_flexion_ROM', 'R_knee_flexion_ROM']

r['pair'] = r.group_a + '  vs  ' + r.group_b
r['n_clin'] = r.group_a.isin(CLINICAL).astype(int) + r.group_b.isin(CLINICAL).astype(int)

# Mixed pairs KEPT here -- that is the point of this variant.

# Order every panel by the knee difference so rows line up and the eye can
# track one pair across all four features. The both-stroke pair is pinned to
# the bottom row (y = 0) rather than sorted in, since it answers a different
# question from the healthy-healthy pairs above it.
key = (r[r.feature == 'L_knee_flexion_ROM']
       .set_index('pair')[['mean_diff', 'n_clin', 'group_a', 'group_b']]
       .sort_values('mean_diff'))


def category(row):
    """Legend category, in the order the user wants them stacked."""
    if row.n_clin == 2:
        return 'stroke'
    if row.n_clin == 1:
        return 'mixed'
    n_tm = (row.group_a in TREADMILL_GROUPS) + (row.group_b in TREADMILL_GROUPS)
    return {0: 'overground', 1: 'modality', 2: 'treadmill'}[n_tm]


# Rows are grouped by category rather than sorted across the whole panel, so
# each legend colour is contiguous. y=0 is the bottom, so the list runs from
# the bottom category upwards: the display order top-to-bottom is
# overground, treadmill, modality, stroke (then mixed, in fig 3b).
CATEGORY_ORDER = ['mixed', 'stroke', 'modality', 'treadmill', 'overground']
CATEGORY_COLOUR = {'overground': CLASS_COLOUR['Healthy (Overground)'],
                   'treadmill': PAIR_TREADMILL,
                   'modality': PAIR_MODALITY,
                   'stroke': CLASS_COLOUR['Stroke (Overground)'],
                   'mixed': PAIR_MIXED}
cat = {p: category(row) for p, row in key.iterrows()}
pairs = [p for c in CATEGORY_ORDER for p in key.index if cat[p] == c]

fig, axes = plt.subplots(1, 4, figsize=(13.0, 8.4), sharey=True)
for ax, f in zip(axes, FEATS):
    d = r[r.feature == f].set_index('pair').reindex(pairs)
    y = np.arange(len(pairs))
    # 10% tint behind each category block, so the groups read as regions
    # rather than only as marker colours.
    for c in CATEGORY_ORDER:
        rows = [i for i, p in enumerate(pairs) if cat[p] == c]
        if rows:
            ax.axhspan(min(rows) - 0.5, max(rows) + 0.5,
                       color=CATEGORY_COLOUR[c], alpha=0.10, lw=0, zorder=0)
    ax.axvline(0, color=INK, lw=1.0, zorder=1)
    for i, (p, row) in enumerate(d.iterrows()):
        n_clin = (row.group_a in CLINICAL) + (row.group_b in CLINICAL)
        # Three kinds of pair, each answering a different question:
        #   0 clinical -> residual site effect between healthy cohorts
        #   1 clinical -> is the clinical signal preserved
        #   2 clinical -> do two independent stroke cohorts agree (harmonisation)
        if n_clin == 2:
            col = CLASS_COLOUR['Stroke (Overground)']
        elif n_clin == 1:
            col = PAIR_MIXED
        else:
            # Healthy-healthy pairs split by capture modality: both treadmill,
            # one of each, or both overground. Display only -- the statistics
            # pool every healthy cohort together.
            n_tm = (row.group_a in TREADMILL_GROUPS) + (row.group_b in TREADMILL_GROUPS)
            col = (PAIR_TREADMILL if n_tm == 2
                   else PAIR_MODALITY if n_tm == 1
                   else CLASS_COLOUR['Healthy (Overground)'])
        sig = row.p_tukey < 0.05
        # Significance by opacity: solid when significant, transparent when not.
        a = 1.0 if sig else 0.3
        ax.plot([row.ci_lo, row.ci_hi], [i, i], color=col, lw=1.6,
                alpha=a, zorder=2)
        # Significant pairs also carry a shape cue, so the distinction
        # survives greyscale printing and colour-vision differences.
        ax.scatter(row.mean_diff, i, s=30 if sig else 22, color=col, zorder=3,
                   alpha=a, marker='^' if sig else 'o',
                   edgecolors='white', linewidths=0.6)
    tidy(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_ylim(-0.8, len(pairs) - 0.2)
    ax.set_title(f.replace('_flexion_ROM', '').replace('_', ' '),
                 fontsize=9.5, color=INK)
    ax.set_xlabel('difference in mean per-trial ROM  (deg)\n95% Tukey-Kramer interval',
                  fontsize=8.5, color=INK)
axes[0].set_yticks(np.arange(len(pairs)))
axes[0].set_yticklabels(pairs, fontsize=6.8, color=INK)

handles = [
    plt.Line2D([], [], marker='o', ls='-', color=CLASS_COLOUR['Healthy (Overground)'],
               label='Healthy (Both Overground)', ms=6),
    plt.Line2D([], [], marker='o', ls='-', color=PAIR_TREADMILL,
               label='Healthy (Both Treadmill)', ms=6),
    plt.Line2D([], [], marker='o', ls='-', color=PAIR_MODALITY,
               label='Healthy (Treadmill vs Overground)', ms=6),
    plt.Line2D([], [], marker='o', ls='-', color=CLASS_COLOUR['Stroke (Overground)'],
               label='Stroke (Both Overground)', ms=6),
    plt.Line2D([], [], marker='o', ls='-', color=PAIR_MIXED,
               label='Stroke vs Healthy', ms=6),
    plt.Line2D([], [], marker='^', ls='-', color=MUTED,
               label='Significant (Tukey-Kramer, a < 0.05)', ms=6),
]
fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
           fontsize=8.5, bbox_to_anchor=(0.5, 1.01))
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = osp.join(HERE, 'fig3b_stroke_vs_healthy.png')
fig.savefig(out, dpi=300, facecolor='white')
print('wrote', out)
