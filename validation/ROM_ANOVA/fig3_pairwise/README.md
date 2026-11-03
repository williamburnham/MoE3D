# Figure 3 — Tukey-Kramer post-hoc differences with 95% intervals

`python fig3_pairwise/make_figure.py` → `fig3_pairwise.png`

Input: `tukey_kramer.csv`.

**What it shows.** All 21 cohort pairs down the y-axis, four features across.
Each row is the difference in mean ROM with its 95% CI. Pairs are coloured by
how many of the two cohorts are clinical, because the three kinds answer
different questions:

| colour | pairs | question |
|---|---|---|
| blue — both healthy | 10 | is there residual site effect after harmonisation? |
| amber — one clinical | 10 | is the clinical signal preserved? |
| red — both clinical | 1 | do two independent stroke cohorts agree? |

Faded rows are not significant after FDR correction within each feature.

**Definition of the plotted quantity.** ROM is the 5th-95th percentile range of
a joint angle *within a single trial*. Each participant's value is the **mean of
their per-trial ROMs**. It is not the range over that participant's pooled
frames, which would be larger, and not a per-gait-cycle ROM, which is the
biomechanical convention and would be slightly smaller.

**What it is for.** It replaces a p-value matrix. A grid of asterisks invites
star-counting and hides the fact that a result can be significant and trivial,
or large and uncertain. A forest plot shows effect size and precision together.

**Why it is drawn this way.**
- Tukey-Kramer: Tukey's HSD extended to unequal group sizes, which is what the
  n range of 9 to 138 requires. The studentized range distribution controls the
  family-wise error rate across all pairs within a feature, so no separate FDR
  correction is applied — that is the test doing its job.
- Assumption to keep in view: Tukey-Kramer pools one within-group variance over
  all groups. Levene rejects homogeneity for most features (SDs span roughly
  5x), so the pooled error term is a compromise between the wide groups
  (10KH, SD 21.9) and the narrow ones (9WRT, SD 4.2). Games-Howell would drop
  that assumption at the cost of a less familiar test.
- Subject means are the unit of analysis. Trials within a participant are not
  independent — measured ICC is 0.92 for G120 and 0.93 for 50ST — so a
  trial-level test inflates F by about 3x.
- Rows are ordered by the left-knee difference and the order is held across all
  four panels, so a single pair can be tracked left to right and L/R
  consistency read off directly.
- Non-significant rows are faded rather than hidden, so absence of evidence is
  visible instead of silently dropped.

**What to look for.** The CI widths carry the real message. `10KH vs 138H`
shows a large knee difference with a CI spanning zero — big effect, 10
participants, not resolvable. `138H vs G120` shows a smaller difference with a
tight CI — modest but real. And `10KS vs 50ST`, the two independent stroke
cohorts, straddles zero on hip and both knees, which is the harmonisation
result worth featuring.

**Caveats.** Two, both worth stating in the methods.

`1MPS` is excluded automatically: it holds one participant (Subj25, the
spastic-paraplegia case), below the minimum for a group. That leaves 9 groups
and 36 pairs per feature.

The three MovePort speed levels contain **the same 24 people**, so pairs within
MovePort are repeated measures, not independent samples. Tukey-Kramer assumes
independence, so those six rows understate the precision available and should
either be read as descriptive or replaced by a paired comparison.
