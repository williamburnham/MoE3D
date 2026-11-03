"""Shared styling so the three figures read as one set."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Keyed by display class: clinical class plus capture modality.
CLASS_COLOUR = {'Healthy (Overground)': '#4C6E8F',
                'Healthy (Treadmill)': '#6FA3B8',
                'Stroke (Overground)': '#C1553B',
                'Stroke (Treadmill)': '#D98A74',
                'Spastic Paraplegia': '#8A6FA8'}

# Cohorts captured on a treadmill. This is a DISPLAY distinction only: they
# remain in the Healthy class for ANOVA and Tukey, so no statistic changes.
TREADMILL_GROUPS = {'9WRT', '24MP'}


def display_class(group, clinical_class):
    """Figure label: clinical class qualified by capture modality.

    Display only -- `clinical_class` in the CSVs is untouched, so every
    healthy cohort still pools together for ANOVA and Tukey.
    """
    if clinical_class in ('Healthy', 'Stroke'):
        mod = 'Treadmill' if group in TREADMILL_GROUPS else 'Overground'
        return f'{clinical_class} ({mod})'
    return clinical_class
PAIR_MIXED = '#C8902F'      # one healthy cohort vs one clinical cohort
# Healthy-healthy pairs split by capture modality, so a residual difference
# between two healthy cohorts can be read as overground/treadmill or not.
PAIR_TREADMILL = CLASS_COLOUR['Healthy (Treadmill)']   # both treadmill
PAIR_MODALITY = '#8A6FA8'                              # treadmill vs overground
BAND = '#E8EDF1'
GRID = '#D9DFE4'
INK = '#2B3539'
MUTED = '#71808A'

# Typical reported ranges for the sagittal range of motion over a gait cycle.
# Used as a visual reference only — attach your own citation before publishing.
NORM_BAND = {'hip': (40, 45), 'knee': (55, 65), 'ankle': (25, 30)}

JOINTS = ['hip', 'knee', 'ankle']
# Display order, set by the user: healthy cohorts first (largest first),
# then the clinical ones. Keeps 50ST and 10KS adjacent for comparison.
# 1MPS last: a single subject, so it reads as an aside rather than a cohort.
GROUP_ORDER = ['138H', 'G120', '10KH', '9WRT', '24MP', '50ST', '10KS', '1MPS']


def tidy(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
