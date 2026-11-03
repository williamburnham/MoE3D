"""Fig 1 — per-cohort ROM distributions."""
import os.path as osp, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from _figstyle import (CLASS_COLOUR, INK, MUTED, JOINTS,
                       GROUP_ORDER, display_class, tidy)

HERE = osp.dirname(osp.abspath(__file__))
t = pd.read_csv(osp.join(HERE, '..', 'participant_rom.csv'))

# Bilateral mean: left/right asymmetry is the subject of figure 2, so collapsing
# the sides here keeps one point per participant and one question per figure.
for j in JOINTS:
    t[j] = t[[f'L_{j}_flexion_ROM', f'R_{j}_flexion_ROM']].mean(axis=1)

# 1MPS is one participant and is excluded from every pairwise test, so it is
# dropped here too. It remains in the CSVs.
DROP = {'1MPS'}
order = [g for g in GROUP_ORDER if g in set(t.group) and g not in DROP]
fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), sharex=True)
rng = np.random.default_rng(0)

for ax, j in zip(axes, JOINTS):
    for i, g in enumerate(order):
        sub = t[t.group == g]
        v = sub[j].dropna().values
        if not len(v):
            continue
        bp = ax.boxplot(v, positions=[i], widths=0.52, showfliers=False,
                        medianprops=dict(color=INK, lw=1.4),
                        boxprops=dict(color=MUTED, lw=0.9),
                        whiskerprops=dict(color=MUTED, lw=0.9),
                        capprops=dict(color=MUTED, lw=0.9))
        for cls, cv in sub.groupby('clinical_class'):
            x = i + rng.uniform(-0.17, 0.17, len(cv))
            ax.scatter(x, cv[j], s=11, alpha=0.65, linewidths=0,
                       color=CLASS_COLOUR.get(display_class(g, cls), MUTED),
                       zorder=3)
        ax.text(i, ax.get_ylim()[0], f'n={len(v)}', ha='center', va='bottom',
                fontsize=6.5, color=MUTED)
    # The three MovePort speeds are the same 24 people walking faster, not three
    # cohorts — shade them so they are not read as independent groups.
    sp = [i for i, g in enumerate(order) if g.startswith('24MP-')
          and False]
    if sp:
        ax.axvspan(min(sp) - 0.45, max(sp) + 0.45, color='#F4F1EA',
                   zorder=-1, lw=0)
    tidy(ax)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha='right', fontsize=8)
    ax.set_title(f'{j} flexion', fontsize=10, color=INK)
    if j == 'hip' and sp:
        ax.text((min(sp) + max(sp)) / 2, ax.get_ylim()[1],
                'MovePort: one cohort, three imposed treadmill speeds',
                ha='center', va='top', fontsize=6.5, color=MUTED)

# Each point is a participant; the value is the mean of that participant's
# per-trial ROMs, not the range over their pooled frames (which would be larger).
axes[0].set_ylabel('mean per-trial ROM  (deg)\none point = one participant',
                   fontsize=9, color=INK)
shown = {display_class(g, c) for g in order
         for c in t[t.group == g].clinical_class.unique()}
handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=k, ms=6)
           for k, c in CLASS_COLOUR.items() if k in shown]
fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
           fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = osp.join(HERE, 'fig1_distributions.png')
fig.savefig(out, dpi=300, facecolor='white')
print('wrote', out)
