# MoE3D — MoE3D: A Dataset of Healthy and Post-Stroke Motion Capture and Electromyography for Machine Learning

MoE3D pairs full-body motion with surface EMG from multiple sub-datasets of healthy,
stroke and spastic-paraplegia participants. Everything is stored once, in its most
complete form, under `data/MoE3D_data/`.
`build_dataset.py` then assembles whichever version of the dataset you need into
`dataset_build/`.


## 1. Quick start

```bash
cd MoE3D
python build_dataset.py --list                       # what is available
python build_dataset.py --motion 263dim_HumanML3D_normalisation \
                        --emg processed --trials all_translated
python build_dataset.py --config config_example.json  # options from a file
```

The second command writes a new folder `dataset_build/<date>_<time>/` holding the
motion files, the EMG cut to match them, Mean/Std, and `build_info.json`.
Requires Python 3 and numpy.


## 2. Folder layout

```
MoE3D/
  build_dataset.py           builds a dataset version (section 5)
  config_example.json        example build options for --config
  README.md                  this file
  dataset_build/             builds are written here
  validation/ROM_ANOVA/      joint range-of-motion ANOVA across sub-datasets
  visualisation/             clip viewer: skeleton and EMG, video export (section 9)
  data/
    MoE3D_data/                      the dataset (not in the git repository)
      motion/
        263dim_HumanML3D_normalisation/{new_joints,new_joint_vecs}/<category>/
        263dim_original_proportions/{new_joints,new_joint_vecs}/<category>/
        SMPLH_AMASS/<category>/
      EMG/
        processed/<emg category>/      one .npy per trial
        raw/<emg category>/            one .npz per trial
        emg_channels.json              channel order and units
      index_2.csv                      clip windows of the original trials
      index_2_translated.csv           clip windows of the translated trials
      trials.csv                       one row per trial (section 6)
      participant_information.csv      sex, age, weight, lesion location, ...
```

The git repository holds the code and documentation only. The data (17 GB) is
distributed separately: put its contents in `data/MoE3D_data/`, so that
`data/MoE3D_data/trials.csv` exists, before running `build_dataset.py`.

Motion categories:

```
gait_overground/
gait_treadmill/original/     gait_treadmill/translated/
other_overground/
other_treadmill/original/    other_treadmill/translated/
```

EMG categories are the same four without the original/translated split: one
recording serves both versions of a treadmill trial.

## 3. Naming

Every file is named

```
{subdataset}-{class}-{action}-{participant}-{repetition}{clip}
```

e.g. `1MPS-SpPa-GaitGr-0025-0100.npy`: subdataset 1MPS, class SpPa (spastic
paraplegia), action GaitGr, participant 0025, repetition 01, clip 00.

| | |
|---|---|
| class | Hlth healthy, Strk stroke, SpPa spastic paraplegia |
| clip | trials longer than 196 frames (9.8 s) are cut into centred, non-overlapping 196-frame clips: 00, 01, 02, ... |
| `M` prefix | `M1MPS-SpPa-GaitGr-0025-0100.npy` is the left/right mirror of that clip (263-dim representations only; EMG is mirrored by swapping left and right channels) |

Trial-level files (SMPLH_AMASS, EMG) have no clip digits:
`1MPS-SpPa-GaitGr-0025-01`. Inside a build, the EMG file always carries exactly
the name of the motion file it belongs to.


## 4. What each representation is

**SMPLH_AMASS** (`.npz`, one per trial, mocap frame rate)
SMPL-H body fitted to the markers by MoSh++ (SOMA pipeline), in AMASS
format: `poses` [T,156] axis-angle, `trans` [T,3] metres (Z up), `betas` [16],
`gender`, `mocap_framerate`. Translated treadmill trials also carry
`treadmill_belt_velocity` [3] (m/s, AMASS frame) and `trim_start_frame` /
`trim_end_frame` (mocap frames of the untrimmed trial; see section 7).
Every file also carries `measured_joints` [52] (bool, SMPL-H joint order, as
in `poses`): False where the joint's rotation came from MoSh++'s pose prior
rather than from markers (see section 7).
The SMPL-H model itself is licensed separately (mano.is.tue.mpg.de).

**263dim_HumanML3D_normalisation** (`.npy`, one per clip, 20 fps)
The HumanML3D motion representation. `new_joint_vecs` [T-1, 263]: root
angular velocity, root linear velocity, root height, joint positions
relative to the root, 6D joint rotations, joint velocities and four foot
contacts. `new_joints` [T-1, 22, 3]: the joint positions recovered from the
vector. Every subject is retargeted onto the skeleton of HumanML3D clip
000021, as HumanML3D does, so body proportions are the same for everyone.

**263dim_original_proportions** (`.npy`, one per clip, 20 fps)
The same representation, but each subject keeps their own skeleton: the
median bone lengths over all their trials, from the fitted SMPL-H body.
Mirrored clips use the mirrored skeleton.

**EMG/processed** (`.npy`, [20 channels, N], 100 Hz, microvolts)
Band-pass 20-450 Hz, full-wave rectified, critically damped low-pass
envelope, resampled to 100 Hz.

**EMG/raw** (`.npz`: `emg` [20, N] float32 microvolts, `fs`, `labels`)
The C3D analog EMG converted to microvolts, unfiltered, at the native
rate (1000, 2000 or 2048 Hz depending on the sub-dataset).

EMG channels (`emg_channels.json`): RERS LERS RRF LRF RVL LVL RBF LBF RST LST
RTA LTA RGAS LGAS RLG LLG RSOL LSOL RIL LIL (erector spinae, rectus femoris,
vastus lateralis, biceps femoris, semitendinosus, tibialis anterior,
gastrocnemius medial head, lateral gastrocnemius, soleus, iliopsoas; R/L).
A channel a sub-dataset did not record is all zeros.


## 5. build_dataset.py

```bash
python build_dataset.py --motion <base> --trials <type> [options]
```

| Option | Values |
|---|---|
| `--motion` | `263dim_original_proportions`<br>`263dim_HumanML3D_normalisation`<br>`SMPLH_AMASS` |
| `--emg` | `processed` (default) \| `raw` \| `none` |
| `--trials` | `overground_gait_only`<br>`original_treadmill_gait_only`<br>`translated_treadmill_gait_only`<br>`overground_and_original_treadmill_gait`<br>`overground_and_translated_treadmill_gait`<br>`all_original`<br>`all_translated` |
| `--subdatasets` | `all` (default) or a list: `--subdatasets 138H 50ST G120` |
| `--exclude` | leave sub-datasets out: `--exclude 1MPS` |
| `--name` | output folder name (default: current date and time, e.g. `2026-11-03_14-05-12`) |
| `--config` | read the options above from a JSON file (see below) |
| `--list` | print the sub-datasets, trial counts and options, then exit |

"original" treadmill trials are as recorded on the belt; "translated" ones are
the same trials re-expressed as overground walking (section 7). The two never
appear in the same build. `all_original` and `all_translated` include every
overground trial plus the treadmill trials in that version.

### Config files

Instead of typing the options, keep them in a JSON file and pass it with
`--config`. The keys are the option names without the dashes: `motion`, `emg`,
`trials`, `subdatasets`, `exclude` and `name`. Any key you leave out takes its
usual default. Keys starting with `_` are ignored, so you can use them as
comments. `config_example.json`:

```json
{
  "_comment": "Options for build_dataset.py --config. Keys starting with _ are ignored. See README.md, section 5.",
  "motion": "263dim_HumanML3D_normalisation",
  "emg": "processed",
  "trials": "overground_and_translated_treadmill_gait",
  "subdatasets": "all",
  "exclude": ["1MPS"],
  "name": "walking_translated_no_1MPS"
}
```

```bash
python build_dataset.py --config config_example.json
python build_dataset.py --config config_example.json --emg raw --name walking_raw_emg
```

Options given on the command line override the file, as in the second line.
`subdatasets` and `exclude` take a list or a single name; `name` may be `null`
for the date and time. Unknown keys and invalid values stop the build before
anything is written. `build_info.json` records the path of the config file and
the options actually used.

Output, `dataset_build/<name>/`:

```
263-dim bases:
  motion/new_joint_vecs/*.npy     the 263-dim vectors
  motion/new_joints/*.npy         the matching joint positions
  Mean.npy, Std.npy               computed over this build (see below)
SMPLH_AMASS:
  motion/*.npz                    one per trial
with EMG:
  emg/*.npy (processed) or emg/*.npz (raw), same names as the motion files
  emg_Mean.npy, emg_Std.npy       per channel, over this build
always:
  build_info.json                 options, counts per sub-dataset, and the
                                  motion files that have no EMG
```

Mean/Std follow HumanML3D: per-dimension mean, one standard deviation shared
within each feature block. Dimensions of joints a sub-dataset never measured
(section 7) are excluded for that sub-dataset. EMG statistics use only channels
that were recorded in each file. Motion files without EMG are kept and listed
under `unpaired_motion` in `build_info.json`.

### Examples

Healthy and stroke level walking, overground only, HumanML3D skeleton:

```bash
python build_dataset.py --motion 263dim_HumanML3D_normalisation \
    --trials overground_gait_only --name walking_overground
```

All walking with treadmill trials as overground, subject skeletons, no
spastic-paraplegia participant:

```bash
python build_dataset.py --motion 263dim_original_proportions \
    --trials overground_and_translated_treadmill_gait --exclude 1MPS
```

Everything, as recorded, with raw EMG, for two sub-datasets:

```bash
python build_dataset.py --motion SMPLH_AMASS --emg raw \
    --trials all_original --subdatasets 24MP 9WRT
```

Motion only:

```bash
python build_dataset.py --motion 263dim_HumanML3D_normalisation \
    --emg none --trials all_translated
```

A build is an independent copy; delete its folder when no longer needed. Time
is dominated by writing files: a full `all_original` build (about 28,000 files,
3 GB) takes around 20 minutes on a USB hard drive.


## 6. trials.csv and the indices

`trials.csv` has one row per trial:

| Column | |
|---|---|
| `trial`, `subdataset`, `clinical_class`, `action`, `participant`, `repetition`, `category` | |
| `mocap_fps` | capture rate of the motion |
| `stage3_step` | every `stage3_step`-th mocap frame becomes one 263-dim frame (blank where the rate is not a multiple of 20) |
| `resampled_to_20fps` | 1 where the motion was interpolated onto the 20 fps grid instead (9WRT, captured at 128 Hz) |
| `motion_fps` | frame rate of the 263-dim files: 20 for every trial |
| `n_mocap_frames` | |
| `imposed_speed` | 24MP treadmill gait: low / medium / high |
| `translated`, `translation_status`, `belt_speed_mps`, `trim_start_frame`, `trim_end_frame` | treadmill translation (section 7) |
| `emg_processed` | 1 if the trial has EMG |
| `emg_raw_fs` | native EMG rate |
| `unmeasured_joints` | SMPL-H body joints with no marker data, `;`-separated (section 7); blank if all were measured |

`index_2.csv` and `index_2_translated.csv` give each clip's window inside its trial
(`trial`, `start_frame`, `end_frame`, `new_name`), in 263-dim frames of the untrimmed
trial. `build_dataset.py` uses them to cut EMG.


## 7. Things to know

### Sub-datasets

| | |
|---|---|
| 138H | healthy, overground, Plug-in Gait |
| 50ST | stroke, overground, Plug-in Gait |
| 24MP | healthy, treadmill (MovePort) |
| 1MPS | one participant with spastic paraplegia (MovePort) |
| G120 | healthy, overground: walking, stairs, slopes, sit-to-stand (Gait120) |
| 9WRT | healthy, split-belt treadmill and exercises |
| 10KH | healthy, overground (Knee-Flex) |
| 10KS | stroke, overground (Knee-Flex) |

### Treadmill translation

On a belt the body stays put and the stance foot slides backwards;
overground the foot is planted and the body travels. Translated trials
remove the belt velocity from the SMPL-H root translation, so they read as
overground walking; every translated 263-dim clip is computed from the
translated SMPL-H. The belt velocity is estimated from the stance feet,
one velocity per trial. MovePort starts the belt with the participant on
it, so trials are trimmed to the stretch where the belt runs at steady
speed (trimmed at the start in 183 trials, median 3.5 s), and a translated
clip may therefore cover a different window from the original clip of the
same name. Two short trials with no steady stretch have no translated
version. Translation changes the reference frame, not the gait: treadmill
walking still differs from overground walking (cadence, stride length,
variability).

### Joints that were never measured

9WRT has no markers on the head, neck or arms; 10KH and 10KS have none above
the pelvis. Those joints come from the body model's pose prior, so they are set
to exactly zero in the 263-dim files (9WRT: neck, head, collars, arms;
10KH/10KS: also the spine).

SMPLH_AMASS keeps the fitted values and marks them instead: `measured_joints`
is False for those joints, and `trials.csv` lists them in `unmeasured_joints`.
They are not zeroed there because a zero rotation is not "missing" in SMPL-H.
It is the rest pose, so the body would show arms held straight out and a rigid
trunk. Leg and pelvis motion does not depend on them: every leg joint descends
from the pelvis, not from the spine. Finger joints (`measured_joints[22:]`) are
False in every file, because no marker set has finger markers.

### Single-leg EMG

G120 and 9WRT recorded the right leg only. Their left channels are zero in
the original files; in mirrored clips the right-leg signal appears on the
left channels.

### EMG amplitude

All EMG is in microvolts, but no MVC normalisation exists, so amplitudes
also reflect each lab's electrodes and amplifiers. Compare amplitudes
between sub-datasets with care.

### Frame rate

Every 263-dim file is exactly 20 fps. Captures at 100 or 200 Hz keep every
5th or 10th frame; 9WRT, captured at 128 Hz, is computed at the full rate
and interpolated onto the 20 fps grid. (HumanML3D's original processing
keeps every `int(fps / 20)`-th frame, which for 128 Hz would give 21.33 fps.)
SMPLH_AMASS keeps each capture's own rate (`mocap_framerate`).

### Body shape

SMPLH_AMASS stores 16 shape coefficients; the joint positions behind both
263-dim bases were computed with the first 10.

### Motion without EMG

Some trials have no EMG recording; their motion is still included.


## 8. Validation

`validation/ROM_ANOVA` compares lower-limb joint range of motion across
sub-datasets (one-way ANOVA, Welch, Tukey-Kramer), from SMPLH_AMASS. The
figures and the summary results are included: `anova_contrasts.csv`,
`tukey_kramer.csv` and `participant_rom.csv`. The figure scripts read the last
two, so the figures can be redrawn without the dataset. To recompute everything
(needs the dataset, numpy, scipy, pandas and matplotlib):

```bash
cd validation/ROM_ANOVA && python main.py
for f in fig*/make_figure.py; do python "$f"; done
```

`main.py` also writes the per-trial tables (`rom_<sub-dataset>.csv`,
`rom_all.csv`, `rom_all_unfiltered.csv`, `anova_trial_level.csv`), which are
not committed.


## 9. Visualisation

`visualisation/visualise_tool.py` plays a clip's skeleton next to its EMG, and
saves what is on screen as a video (`.mp4`, `.gif`) or as a still figure of
overlaid frames. Needs numpy and matplotlib, tkinter for the window, and ffmpeg
for `.mp4`.

```bash
cd visualisation
python visualise_tool.py                                  # open the viewer, add files
python visualise_tool.py 138H-Hlth-GaitGr-0001-0100       # one clip by name
python visualise_tool.py 24MP-Hlth-GaitTm-0013-05 --source hml3d-tt   # all clips of a trial
python visualise_tool.py 138H-Hlth-GaitGr-0001-01 --source amass      # a whole SMPL-H trial
python visualise_tool.py 138H-Hlth-GaitGr-0001-0100 --save clip.mp4   # no window
python visualise_tool.py 138H-Hlth-GaitGr-0001-0100 --save-figure clip.png
```

**Add files…** Files can be picked from `new_joints` or `new_joint_vecs` (the matching joint
positions are shown), from `SMPLH_AMASS`, and also from a build in
`dataset_build/`, in which case the build's own `emg/` files are used.

| `--source` | joints shown when a clip is named rather than picked |
|---|---|
| `hml3d` (default) | 263dim_HumanML3D_normalisation |
| `hml3d-tt` | the same, with treadmill trials translated |
| `subject` | 263dim_original_proportions |
| `subject-tt` | the same, with treadmill trials translated |
| `amass` | SMPLH_AMASS, one whole trial per file |
| `amass-tt` | the same, with treadmill trials translated |

Overground clips are the same in the original and translated sources. Naming a
263-dim clip with an `amass` source opens its trial, and naming a trial with a
263-dim source opens all of its clips; the **Joints source** buttons in the
window switch the same way. Mirrored clips (`M` prefix) exist in 263-dim only.

Unmeasured joints are not drawn (in SMPLH_AMASS files, those marked False in
`measured_joints`), and unrecorded EMG channels are greyed out.
`--emg-scale muscle` scales each left/right pair together, so
the two sides can be compared; `--global-frame` fixes the camera so the subject
is seen travelling. `python visualise_tool.py --help` lists every option.

`--body` (or **SMPL body** in the window) overlays the neutral SMPL-H body
fitted to the joints. It needs the SMPL-H body model, as do exact skeletons for
SMPLH_AMASS files (see below).

### Installing the SMPL-H body model (optional)

The SMPL-H model is licensed separately and is not part of MoE3D. Everything
except `--body` and exact SMPLH_AMASS skeletons works without it.

1. Register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) and accept the
   licence.
2. On the **Download** page, download **Extended SMPL+H model** (the version
   used by AMASS), `smplh.tar.xz`.
3. Unpack it into `visualisation/body_models/smplh/`:

   ```bash
   mkdir -p visualisation/body_models/smplh
   tar -xf smplh.tar.xz -C visualisation/body_models/smplh
   ```

4. Check that these three files exist in the correct locations:

   ```
   visualisation/body_models/smplh/neutral/model.npz
   visualisation/body_models/smplh/male/model.npz
   visualisation/body_models/smplh/female/model.npz
   ```
