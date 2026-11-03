"""Fig 2 — left vs right ROM per participant; distance from the diagonal is asymmetry."""
import os.path as osp, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from _figstyle import (CLASS_COLOUR, INK, MUTED, GRID, JOINTS,
                       display_class, tidy)

HERE = osp.dirname(osp.abspath(__file__))
t = pd.read_csv(osp.join(HERE, '..', 'participant_rom.csv'))
# Spastic Paraplegia is a single participant and is excluded from every
# pairwise test, so it is dropped here too. The CSVs keep it.
t = t[t.clinical_class != 'Spastic Paraplegia']
t['display_class'] = [display_class(g, c)
                      for g, c in zip(t.group, t.clinical_class)]

fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.6))
for ax, j in zip(axes, JOINTS):
    L, R = f'L_{j}_flexion_ROM', f'R_{j}_flexion_ROM'
    d = t[[L, R, 'display_class', 'group', 'subject']].dropna()
    lim = (0, max(d[L].max(), d[R].max()) * 1.06)
    # Identity line = perfect symmetry; distance from it is the asymmetry.
    ax.plot(lim, lim, color=INK, lw=1.0, zorder=1)
    for cls, cv in d.groupby('display_class'):
        ax.scatter(cv[L], cv[R], s=16, alpha=0.7, linewidths=0,
                   color=CLASS_COLOUR.get(cls, MUTED), zorder=3, label=cls)
    tidy(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_title(f'{j} flexion', fontsize=10, color=INK)
    ax.set_xlabel('left — mean per-trial ROM  (deg)', fontsize=8.5, color=INK)
axes[0].set_ylabel('right — mean per-trial ROM  (deg)',
                   fontsize=8.5, color=INK)
shown = set(t.display_class)
handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=k, ms=6)
           for k, c in CLASS_COLOUR.items() if k in shown]
fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
           fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0.02, 1, 0.93))
out = osp.join(HERE, 'fig2_asymmetry.png')
# bbox_inches='tight' so the axis labels cannot be clipped: the panels
# use equal aspect, so their height is set by the data range rather than
# by the figure size, and tight_layout alone does not always leave room.
fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight',
            pad_inches=0.12)
print('wrote', out)
