# Figure 2 — left versus right ROM; distance from the diagonal is asymmetry

`python fig2_asymmetry/make_figure.py` → `fig2_asymmetry.png`

**What it shows.** One point per participant: left-side ROM on the x-axis,
right-side on the y. The solid line is perfect symmetry; dashed guides mark
±25%. Colour is clinical class.

**Definition of the plotted quantity.** ROM is the 5th-95th percentile range of
a joint angle *within a single trial*. Each participant's value is the **mean of
their per-trial ROMs**. It is not the range over that participant's pooled
frames, which would be larger, and not a per-gait-cycle ROM, which is the
biomechanical convention and would be slightly smaller.

**What it is for.** This is the strongest single piece of evidence that the
pipeline preserved pathology. Inter-limb asymmetry is the defining feature of
hemiplegic gait, and it is a property the processing could easily have
destroyed — `uniform_skeleton` retargets every subject onto one template, and
the mirror augmentation reflects clips across the sagittal plane. If asymmetry
survived both, the clinical signal is intact.

It also uses the bilateral structure of the data rather than averaging it away,
which figure 1 necessarily does.

**Why it is drawn this way.**
- The identity line turns a two-number comparison into a one-dimensional read:
  how far from the line, and which side.
- Magnitude and asymmetry appear together. A participant low-left on the plot
  has reduced ROM bilaterally; one far off the diagonal has a side difference.
  These are clinically distinct and a box plot cannot separate them.
- The three most asymmetric participants per joint are labelled, so the rows
  worth opening individually are named rather than left to be hunted for.

**What to look for.** Healthy participants form a tight diagonal cloud. Stroke
participants are shifted toward the origin (lower ROM) and scatter off the
line. `50ST-TVC03` is the clearest hemiplegic case, at roughly 56° left hip
against 14° right.

**Caveat.** Some labelled outliers are worth checking before they are used as
examples — `10KH-S06` and `24MP-Subj4` sit far off the diagonal on ankle but
are healthy participants, so those are more likely fit artefacts than biology.
