# Figure 1 — ROM distributions per cohort, against typical gait ranges

`python fig1_distributions/make_figure.py` → `fig1_distributions.png`

**What it shows.** Hip, knee and ankle flexion ROM for all 362 participants,
one dot each, grouped by sub-dataset. Box = median and quartiles. Dots are
coloured by clinical class. The shaded band is the typical reported range for
that joint over a gait cycle.

**Definition of the plotted quantity.** ROM is the 5th-95th percentile range of
a joint angle *within a single trial*. Each participant's value is the **mean of
their per-trial ROMs**. It is not the range over that participant's pooled
frames, which would be larger, and not a per-gait-cycle ROM, which is the
biomechanical convention and would be slightly smaller.

**What it is for.** This is the *external validity* panel: it answers "does
MoE3D contain plausible human gait?" rather than "do the cohorts differ". A
reader can check in one glance whether each cohort sits where human walking
should sit — which no amount of between-group testing can establish.

**Why it is drawn this way.**
- Every participant is a dot. With n=9 (9WRT) and n=10 (Knee-Flex), a violin or
  smooth density would imply precision that is not there. The dots also make
  10KH's very wide spread visible, which matters because it is what makes that
  group's comparisons so uncertain.
- Left and right are averaged. Asymmetry is figure 2's job; collapsing sides
  here keeps one participant to one dot and one question to one figure.
- Colour encodes clinical class, not group. Seven colours would be unreadable;
  three is immediate.
- `n=` is printed under each cohort because the group sizes span 9 to 138 and
  the boxes do not convey that.

**What to look for.** The healthy cohorts cluster near the typical band for all
three joints. Both stroke cohorts sit below it on knee and ankle. G120 runs low
on knee and high on ankle relative to the other healthy groups; 10KH runs high
on knee. Those two are the drivers of the significant healthy-only ANOVA.

**Caveat.** The band is a visual reference drawn from commonly quoted gait
values, not a citation — attach your own source before publishing, and note
that ROM here is computed over a whole trial rather than a single gait cycle,
which inflates it slightly relative to textbook figures.
