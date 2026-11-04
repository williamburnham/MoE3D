#!/usr/bin/env python3
"""
visualise_tool.py — play and export MoE3D clips.

Shows a clip's 22-joint skeleton and its 20-channel EMG side by side, either
panel independently toggleable, and exports what is on screen to a video file
or a still figure.

Point it at a clip or trial name, a file from data/MoE3D_data/motion/ (263-dim
.npy or SMPLH_AMASS .npz) or from a build in dataset_build/, or nothing at all
and pick files in the GUI. Motion and
EMG are paired by name: the EMG is cut from the per-trial recording with the
clip indices, exactly as build_dataset.py does. For a file picked from a build,
the build's own emg/ folder is used.

Usage
-----
  python visualise_tool.py                          open empty, pick files in the GUI
  python visualise_tool.py 138H-Hlth-GaitGr-0001-0100      open one clip by name
  python visualise_tool.py 24MP-Hlth-GaitTm-0013-05        a trial prefix opens all its clips
  python visualise_tool.py --source subject <name>          each subject's own skeleton
  python visualise_tool.py --source hml3d-tt <name>         treadmill trials translated
  python visualise_tool.py --source amass 138H-Hlth-GaitGr-0001-01   a whole SMPL-H trial
  python visualise_tool.py ../dataset_build/<build>/motion/new_joints/*.npy
  python visualise_tool.py <name> --save out.mp4    render headlessly, no GUI
  python visualise_tool.py <name> --save out.mp4 --no-emg --fps 20
  python visualise_tool.py <name> --global-frame     watch them travel, not centred
  python visualise_tool.py <name> --emg-scale muscle compare left vs right strength
  python visualise_tool.py <name> --body --smplh-model /path/to/smplh/neutral/model.npz

The MoE3D folder defaults to this script's parent directory; override with
--data-root.

Author: Billy Burnham (2026)
"""

import argparse
import os
import os.path as osp
import sys
from glob import glob

import numpy as np

# HumanML3D 22-joint skeleton (paramUtil.t2m_kinematic_chain)
CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
         [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
# limb colouring: right leg, left leg, spine/head, right arm, left arm
CHAIN_SIDE = ['R', 'L', 'C', 'R', 'L']

# The four joint sources. Keys are what --source accepts; values are the
# motion base under data/MoE3D_data/motion/ and the treadmill version.
#   hml3d     retargeted onto the HumanML3D template skeleton
#   subject   each subject's own skeleton
#   *-tt      treadmill trials translated so the belt is cancelled and the
#             subject travels; overground clips are the same in both versions
SOURCES = {
    'hml3d':      ('263dim_HumanML3D_normalisation', 'original'),
    'hml3d-tt':   ('263dim_HumanML3D_normalisation', 'translated'),
    'subject':    ('263dim_original_proportions', 'original'),
    'subject-tt': ('263dim_original_proportions', 'translated'),
    'amass':      ('SMPLH_AMASS', 'original'),
    'amass-tt':   ('SMPLH_AMASS', 'translated'),
}
SOURCE_LABELS = {
    'hml3d':      'HumanML3D skeleton',
    'hml3d-tt':   'HumanML3D, translated',
    'subject':    'Subject skeleton',
    'subject-tt': 'Subject, translated',
    'amass':      'SMPL-H (AMASS)',
    'amass-tt':   'SMPL-H, translated',
}
AMASS = 'SMPLH_AMASS'

# SMPL-H body joints 0-21 are the HumanML3D 22 joints, in the same order.
SMPLH_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                 16, 17, 18, 19]
# Rest-pose joints of the neutral SMPL-H body with zero shape, used only when
# the SMPL-H model is not installed: the skeleton then has template
# proportions instead of the subject's (betas need the model).
SMPLH_TEMPLATE_J = np.array([
    [-0.0018, -0.2233, 0.0282], [0.0677, -0.3147, 0.0214],
    [-0.0695, -0.3139, 0.0239], [-0.0043, -0.1144, 0.0015],
    [0.1020, -0.6899, 0.0169], [-0.1078, -0.6964, 0.0150],
    [0.0012, 0.0208, 0.0026], [0.0884, -1.0879, -0.0268],
    [-0.0920, -1.0948, -0.0273], [0.0026, 0.0737, 0.0280],
    [0.1148, -1.1437, 0.0925], [-0.1174, -1.1430, 0.0961],
    [-0.0002, 0.2876, -0.0148], [0.0815, 0.1955, -0.0060],
    [-0.0791, 0.1926, -0.0106], [0.0050, 0.3526, 0.0365],
    [0.1724, 0.2260, -0.0149], [-0.1752, 0.2251, -0.0197],
    [0.4321, 0.2132, -0.0424], [-0.4289, 0.2118, -0.0411],
    [0.6813, 0.2222, -0.0435], [-0.6842, 0.2196, -0.0467]])
# The file dialog opens here until a file has been picked.
DEFAULT_PICK_DIR = '263dim_HumanML3D_normalisation'

# Channel order of data/MoE3D_data/EMG/emg_channels.json.
EMG_LABELS = ['RERS', 'LERS', 'RRF', 'LRF', 'RVL', 'LVL', 'RBF', 'LBF',
              'RST', 'LST', 'RTA', 'LTA', 'RGAS', 'LGAS', 'RLG', 'LLG',
              'RSOL', 'LSOL', 'RIL', 'LIL']

COL = {'R': '#D8524A', 'L': '#1F9E86', 'C': '#6C7F8A'}
FPS_DEFAULT = 20            # joints are stored at 20 fps

# Mesh-overlay defaults. BODY_DETAIL is a vertex-clustering cell size in metres:
# the full 13,776-triangle SMPL body takes ~460 ms per frame to draw in
# mplot3d, which is unusable for playback, so the mesh is decimated. 0.035 m
# leaves ~3,000 faces (~1,500 after back-face culling) and runs near 14 fps.
BODY_DETAIL = 0.035
# Same light grey as the EMG-placement and sample figures, so the
# visualiser and the paper figures show the same body.
BODY_RGB = np.array([0.62, 0.63, 0.65])
BODY_LIGHT = np.array([0.35, -0.72, 0.60])     # axes coordinates, see to_axes
BODY_LIGHT = BODY_LIGHT / np.linalg.norm(BODY_LIGHT)
# Camera azimuth in axes coordinates: behind the subject and to its left.
VIEW_AZIM = 25


def to_axes(P):
    """World (x, y, z), Y-up, -> mplot3d axes (x, -z, y), whose third axis is up.

    The negation matters: swapping y and z alone is a reflection, and drew
    every clip as its mirror image (the subject's left limbs on its right).
    """
    P = np.asarray(P)
    return np.stack([P[..., 0], -P[..., 2], P[..., 1]], -1)


# ── data ─────────────────────────────────────────────────────────────────────

class Clip:
    """One clip: joints (T,22,3) and/or EMG (20,N). Either may be absent."""

    def __init__(self, stem, joints, emg, joints_src, emg_src, emg_rate=100.0,
                 label=''):
        self.stem = stem
        self.joints = joints
        self.emg = emg
        self.joints_src = joints_src
        self.emg_src = emg_src
        self.emg_rate = emg_rate
        self.label = label

    @property
    def n_frames(self):
        if self.joints is not None:
            return len(self.joints)
        return max(1, int(round(self.emg.shape[1] / self.emg_rate * FPS_DEFAULT)))

    def emg_at(self, frame):
        """EMG sample index for a motion frame, however the two lengths differ."""
        if self.emg is None:
            return 0
        return int(round(frame / max(1, self.n_frames - 1) * (self.emg.shape[1] - 1)))

    def describe(self):
        bits = []
        if self.label:
            bits.append(self.label)
        bits.append(f"{self.n_frames} frames")
        if self.joints is not None:
            bits.append("motion")
        if self.emg is not None:
            live = int((np.abs(self.emg).sum(1) > 0).sum())
            bits.append(f"EMG {live}/{self.emg.shape[0]} ch")
        return "  ·  ".join(bits)


class Release:
    """Finds clips in a MoE3D folder and cuts their EMG.

    Motion lives in data/MoE3D_data/motion/<base>/new_joints/<category>/, with
    treadmill categories split into original/ and translated/. EMG is stored
    per trial and cut to each clip with the clip indices; build_dataset.py's
    EMGSource does that, so the viewer and the builds cut it identically.
    """

    def __init__(self, root):
        self.root = root
        self.data = osp.join(root, 'data', 'MoE3D_data')
        self.motion = osp.join(self.data, 'motion')
        self._index = {}
        self._emg = None
        self._emg_error = None

    def ok(self):
        return osp.isdir(self.motion)

    def pick_dir(self):
        d = osp.join(self.motion, DEFAULT_PICK_DIR)
        return d if osp.isdir(d) else self.root

    def index(self, base):
        """{stem: {version: path}} for one motion base; version None = overground."""
        if base not in self._index:
            out = {}
            top, ext = ((osp.join(self.motion, base), '.npz') if base == AMASS
                        else (osp.join(self.motion, base, 'new_joints'), '.npy'))
            for dirpath, _, files in os.walk(top):
                leaf = osp.basename(dirpath)
                version = leaf if leaf in ('original', 'translated') else None
                for f in files:
                    if f.endswith(ext):
                        out.setdefault(f[:-4], {})[version] = osp.join(dirpath, f)
            self._index[base] = out
        return self._index[base]

    def find(self, stem, source):
        """(path, version) of a clip; overground clips exist in one version only."""
        base, version = SOURCES[source]
        hit = self.index(base).get(as_trial(stem, self.index(base)) if base == AMASS
                                   else stem, {})
        if version in hit:
            return hit[version], version
        return hit.get(None), 'original'

    def emg_source(self):
        if self._emg is None and self._emg_error is None:
            try:
                sys.path.insert(0, self.root)
                import build_dataset
                if osp.abspath(build_dataset.DATA) != osp.abspath(self.data):
                    raise RuntimeError('build_dataset.py found is not the one in %s'
                                       % self.root)
                self._emg = build_dataset.EMGSource('processed')
            except Exception as exc:              # no EMG folder, etc.
                self._emg_error = exc
                print(f'EMG unavailable: {exc}', file=sys.stderr)
        return self._emg

    def emg_for(self, stem, version, trial=False):
        """EMG cut to a 263-dim clip, or a whole trial's EMG for SMPLH_AMASS."""
        src = self.emg_source()
        if src is None:
            return None
        got = src.for_trial(stem, version) if trial else src.for_clip(stem, version)
        if got is None:
            return None
        x, rate = got
        return x.astype(np.float64), rate


def as_trial(stem, idx):
    """A 263-dim clip name to its SMPLH_AMASS trial (two clip digits dropped).

    Mirrored clips have no SMPL-H trial: the mirror exists only in 263-dim.
    """
    if stem in idx or stem.startswith('M'):
        return stem
    return stem[:-2] if stem[:-2] in idx else stem


# ── SMPL-H forward kinematics ────────────────────────────────────────────────

_SMPLH = {}


def smplh_rest_joints(gender, betas):
    """(22,3) rest joints for this body and whether they are exact.

    Uses the gendered SMPL-H model next to smpl_overlay.MODEL (falling back to
    neutral). Without any model, returns the zero-shape template.
    """
    import smpl_overlay
    models = osp.dirname(osp.dirname(smpl_overlay.MODEL))
    for g in (gender, 'neutral'):
        path = osp.join(models, g, 'model.npz')
        if g == 'neutral' and not osp.isfile(path):
            path = smpl_overlay.MODEL
        if not osp.isfile(path):
            continue
        if path not in _SMPLH:
            d = np.load(path, allow_pickle=True)
            _SMPLH[path] = (np.asarray(d['J_regressor'], np.float64)[:22],
                            np.asarray(d['v_template'], np.float64),
                            np.asarray(d['shapedirs'], np.float64))
        R, V, S = _SMPLH[path]
        n = min(len(betas), S.shape[2])
        return R.dot(V + S[:, :, :n].dot(betas[:n])), g
    return SMPLH_TEMPLATE_J.copy(), None


def _rodrigues(aa):
    """(...,3) axis-angle to (...,3,3) rotation matrices."""
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-12)
    x, y, z = k[..., 0], k[..., 1], k[..., 2]
    O = np.zeros_like(x)
    K = np.stack([O, -z, y, z, O, -x, -y, x, O], -1).reshape(aa.shape[:-1] + (3, 3))
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def amass_joints(path):
    """22 joints (T,3) at 20 fps, Y up, from an SMPLH_AMASS file; and a label."""
    z = np.load(path)
    poses, trans = z['poses'].astype(np.float64), z['trans'].astype(np.float64)
    fps = float(z['mocap_framerate'])
    J0, model = smplh_rest_joints(str(z['gender']), z['betas'].astype(np.float64))
    R = _rodrigues(poses[:, :66].reshape(-1, 22, 3))
    G = np.empty_like(R)
    P = np.empty((len(poses), 22, 3))
    G[:, 0], P[:, 0] = R[:, 0], J0[0] + trans
    for j in range(1, 22):
        p = SMPLH_PARENTS[j]
        G[:, j] = G[:, p] @ R[:, j]
        P[:, j] = P[:, p] + G[:, p] @ (J0[j] - J0[p])
    # AMASS is Z up; the viewer, like HumanML3D, is Y up. This is HumanML3D's
    # own conversion (swap y and z, then negate x: together a rotation), so
    # these trials are drawn exactly like the 263-dim clips made from them.
    P = np.stack([-P[..., 0], P[..., 2], P[..., 1]], -1)
    # Resample to the viewer's 20 fps (captures are 100, 128 or 200 Hz).
    t = np.arange(len(P)) / fps
    tq = np.arange(0.0, t[-1] + 1e-9, 1.0 / FPS_DEFAULT)
    flat = P.reshape(len(P), -1)
    P = np.stack([np.interp(tq, t, flat[:, c]) for c in range(flat.shape[1])],
                 1).reshape(len(tq), 22, 3)
    if 'measured_joints' in z.files:
        # Joints MoSh++ filled from its pose prior are zeroed, as in the
        # 263-dim files, so they are left out rather than drawn as a guess.
        P[:, ~z['measured_joints'][:22].astype(bool)] = 0.0
    label =('SMPL-H %s model' % model) if model else \
        'approximate skeleton (no SMPL-H model)'
    return P, label


def _stem(token):
    stem = osp.basename(token)
    for ext in ('.npy', '.npz'):
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
    return stem


def resolve_stems(token, rel, source='hml3d'):
    """Expand a token into clip stems.

    An existing file or an exact stem wins. Otherwise the token is treated as a
    trial-name prefix, so '24MP-Hlth-GaitTm-0013-05' finds its clips 00, 01, ...
    — long trials are cut into 196-frame clips and nobody remembers the digits.
    """
    if osp.isfile(token):
        return [token]
    stem = _stem(token)
    idx = rel.index(SOURCES[source][0]) if rel.ok() else {}
    if SOURCES[source][0] == AMASS:
        stem = as_trial(stem, idx)
    if stem in idx:
        return [stem]
    return sorted(k for k in idx if k.startswith(stem))


def _build_paths(path):
    """For a file inside dataset_build/<name>/motion/: (joints path, EMG path)."""
    parts = osp.abspath(path).split(osp.sep)
    if 'motion' not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index('motion')
    build = osp.sep.join(parts[:i])
    if not osp.isfile(osp.join(build, 'build_info.json')):
        return None
    stem = _stem(path)
    jpath = (osp.abspath(path) if path.endswith('.npz')
             else osp.join(build, 'motion', 'new_joints', stem + '.npy'))
    epath = None
    for ext in ('.npy', '.npz'):
        if osp.isfile(osp.join(build, 'emg', stem + ext)):
            epath = osp.join(build, 'emg', stem + ext)
    return jpath, epath, osp.basename(build)


def _load_emg_file(path):
    if path.endswith('.npz'):
        z = np.load(path)
        return z['emg'].astype(np.float64), float(z['fs'])
    return np.load(path).astype(np.float64), 100.0


def load_clip(token, rel, source='hml3d'):
    """Resolve a stem or path into a Clip, pairing motion and EMG by name."""
    stem = _stem(token)
    base, version = SOURCES[source]
    jpath = epath = None
    emg, rate, label = None, 100.0, SOURCE_LABELS[source]

    if osp.isfile(token):
        # An explicit file wins over the selected source: picking a translated
        # clip must not silently load its untranslated twin of the same name.
        built = _build_paths(token)
        if built:
            jpath, epath, bname = built
            label = 'build ' + bname
            if epath:
                emg, rate = _load_emg_file(epath)
        else:
            parts = osp.abspath(token).split(osp.sep)
            jpath = osp.abspath(token)
            if 'new_joint_vecs' in parts:     # the vector's matching positions
                k = len(parts) - 1 - parts[::-1].index('new_joint_vecs')
                jpath = osp.sep.join(parts[:k] + ['new_joints'] + parts[k + 1:])
            leaf = parts[-2]
            version = leaf if leaf in ('original', 'translated') else 'original'
            base = (AMASS if token.endswith('.npz') else
                    next((p for p in parts if p.startswith('263dim_')), base))
            label = '%s%s' % (base.replace('263dim_', ''),
                              ', translated' if version == 'translated' else '')
    else:
        if rel.ok():
            jpath, version = rel.find(stem, source)
            if jpath:
                stem = _stem(jpath)

    is_amass = bool(jpath) and jpath.endswith('.npz')
    if jpath is None and base == AMASS:
        raise FileNotFoundError(
            f"no SMPLH_AMASS trial for '{stem}'"
            + (" (mirrored clips exist in 263-dim only)" if stem.startswith('M') else ''))
    if epath is None:
        got = rel.emg_for(stem, version, trial=is_amass)
        if got is not None:
            emg, rate = got
            epath = 'data/MoE3D_data/EMG/processed'

    joints = None
    if is_amass and osp.exists(jpath):
        joints, how = amass_joints(jpath)
        label += ', ' + how
    elif jpath and osp.exists(jpath):
        j = np.load(jpath)
        if j.ndim == 3 and j.shape[1] >= 22:
            joints = j[:, :22, :].astype(np.float64)
    else:
        jpath = None

    if joints is None and emg is None:
        raise FileNotFoundError(
            f"no motion or EMG found for '{stem}' in {SOURCE_LABELS[source]}.\n"
            f"  looked under {osp.join(rel.motion, base)}")
    return Clip(stem, joints, emg, jpath, epath, rate, label)


# ── figure ───────────────────────────────────────────────────────────────────

def build_figure(clip, show_motion, show_emg, figsize=(11, 5.2), dpi=100,
                 follow=True, emg_scale='channel', show_body=False,
                 body_detail=BODY_DETAIL, show_skeleton=True):
    """Return (fig, draw(frame)). Panels present depend on the toggles.

    follow=True keeps the camera locked on the root, so the body fills the panel
    and gait is easy to read. follow=False fixes the camera over the whole
    trajectory instead, so the subject is seen travelling through the global
    frame — smaller on screen, but the translation is visible.

    emg_scale='channel' normalises every trace to its own peak, which makes a
    quiet muscle as readable as a loud one but hides any left/right difference.
    emg_scale='muscle' normalises each L/R pair to their shared peak instead, so
    the two sides of a muscle can be compared by eye — a weaker side stays
    visibly shorter. Where only one side was recorded the unrecorded trace sits
    flat on its baseline, which is itself the honest picture.
    """
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D            # noqa: F401 (registers 3d)
    import matplotlib.pyplot as plt

    show_motion = show_motion and clip.joints is not None
    show_emg = show_emg and clip.emg is not None
    if not show_motion and not show_emg:
        raise ValueError("nothing to draw: both panels are off or unavailable")
    if show_motion and not show_skeleton and not show_body:
        raise ValueError("motion panel would be empty: enable the skeleton or "
                         "the SMPL body")

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')
    ax3d = ax_emg = None
    if show_motion and show_emg:
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax_emg = fig.add_subplot(1, 2, 2)
    elif show_motion:
        ax3d = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        ax_emg = fig.add_subplot(1, 1, 1)

    lines3d = []
    floor = None
    body_pc = None
    # Joints a cohort never measured are stored as exact zeros in every
    # representation. Treat them as absent rather than drawing limbs to the
    # world origin; the data states what is missing, the viewer respects it.
    missing = set()
    if clip.joints is not None:
        J22 = clip.joints[:, :22]
        missing = {int(i) for i in range(22)
                   if np.abs(J22[:, i]).max() == 0.0 and i != 0}
    if ax3d is not None:
        J = clip.joints
        # Box sized to the body, not the whole travelled path, and re-centred on
        # the root each frame — otherwise a walking clip zooms out to a speck.
        lo, hi = J.reshape(-1, 3).min(0), J.reshape(-1, 3).max(0)
        stature = float(hi[1] - lo[1])
        span = max(stature, 0.9) * 0.52
        ground = float(lo[1])
        # Global frame: one fixed box around the whole path. The horizontal
        # extents are squared up so the ground plane is not sheared, and the
        # box aspect follows the true proportions so the body is not stretched.
        if not follow:
            pad = stature * 0.35
            x0, x1 = float(lo[0]) - pad, float(hi[0]) + pad
            z0, z1 = float(lo[2]) - pad, float(hi[2]) + pad
            half = max(x1 - x0, z1 - z0) / 2.0
            xc, zc = (x0 + x1) / 2.0, (z0 + z1) / 2.0
            gx0, gx1 = xc - half, xc + half
            gz0, gz1 = zc - half, zc + half
            gy1 = ground + max(stature * 1.15, 2 * half * 0.45)
            ax3d.set_xlim(gx0, gx1)
            ax3d.set_ylim(-gz1, -gz0)
            ax3d.set_zlim(ground, gy1)
        if follow:
            ax3d.set_box_aspect((1, 1, 1))
        else:
            ax3d.set_box_aspect((2 * half, 2 * half, gy1 - ground))
        ax3d.set_axis_off()
        # Rear-quarter view from the subject's left (HumanML3D clips start
        # facing +z). Viewed more side-on, the left and right limbs project to
        # almost the same screen x and cannot be told apart by eye; at this
        # angle they separate. Drag the panel to rotate.
        ax3d.view_init(elev=12, azim=VIEW_AZIM)
        floor, = ax3d.plot([], [], [], color='#DCE3E6', lw=1.0, zorder=0)
        if show_skeleton:
            for side in CHAIN_SIDE:
                ln, = ax3d.plot([], [], [], '-o', lw=3.0, ms=4.0,
                                color=COL[side], solid_capstyle='round')
                lines3d.append(ln)
        body_pc = None
        if show_body:
            try:
                import smpl_overlay as _so
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                body_model = _so.load_model(detail=body_detail)
                body_keep = _so.visible_faces_mask(body_model, missing)
                # No edge stroke: at this decimation the triangle outlines
                # dominate and the body reads as a wire cage rather than skin.
                body_pc = Poly3DCollection([], edgecolor='none',
                                           alpha=0.55, zorder=1)
                ax3d.add_collection3d(body_pc)
            except Exception as exc:                  # model missing, etc.
                body_pc = None
                print(f'body overlay unavailable: {exc}')
        if show_skeleton:
            ax3d.text2D(0.02, 0.97, 'right limb', transform=ax3d.transAxes,
                        fontsize=8, color=COL['R'], family='monospace', va='top')
            ax3d.text2D(0.02, 0.93, 'left limb', transform=ax3d.transAxes,
                        fontsize=8, color=COL['L'], family='monospace', va='top')
        ax3d.text2D(0.02, 0.89,
                    'rear-quarter · ' + ('follows subject' if follow else 'global frame'),
                    transform=ax3d.transAxes,
                    fontsize=7, color='#9AA7AD', family='monospace', va='top')

    playhead = None
    if ax_emg is not None:
        E = clip.emg
        live = np.abs(E).sum(1) > 0
        n = E.shape[1]
        t = np.arange(n)
        # EMG_LABELS pairs each muscle as (R, L) at indices (2k, 2k+1), so a
        # shared scale is just the max over each adjacent pair.
        peaks = np.abs(E).max(axis=1)
        if emg_scale == 'muscle':
            pair = np.maximum(peaks[0::2], peaks[1::2])
            scale = np.repeat(pair, 2)[:E.shape[0]]
        else:
            scale = peaks
        scale = np.where(scale > 0, scale, 1.0)
        for i in range(E.shape[0]):
            y_off = E.shape[0] - 1 - i
            side = 'R' if EMG_LABELS[i][0] == 'R' else 'L'
            c = COL[side] if live[i] else '#C9D2D6'
            ax_emg.plot(t, y_off + 0.78 * np.abs(E[i]) / scale[i], lw=.75, color=c)
            ax_emg.text(-n * 0.015, y_off, EMG_LABELS[i], ha='right', va='center',
                        fontsize=6.5, family='monospace',
                        color='#33424A' if live[i] else '#AEBCC2')
        ax_emg.set_xlim(0, n - 1)
        ax_emg.set_ylim(-0.6, E.shape[0])
        ax_emg.set_yticks([])
        ax_emg.set_xticks([])
        for s in ax_emg.spines.values():
            s.set_visible(False)
        ax_emg.set_title(
            'EMG  ·  rectified, ' + ("each channel to its muscle's peak (L/R comparable)"
                                     if emg_scale == 'muscle'
                                     else 'each channel to its own peak'),
            fontsize=8, color='#55636A', loc='left')
        playhead = ax_emg.axvline(0, color='#C8811F', lw=1.4)

    title = fig.suptitle('', fontsize=9.5, family='monospace', color='#33424A')

    def draw(frame):
        frame = int(np.clip(frame, 0, clip.n_frames - 1))
        if ax3d is not None:
            P = clip.joints[frame]
            if follow:
                cx, cz = float(P[0, 0]), float(P[0, 2])
                ax3d.set_xlim(cx - span, cx + span)
                ax3d.set_ylim(-cz - span, -cz + span)
                ax3d.set_zlim(ground, ground + 2 * span)
                r = span * 0.9
                floor.set_data([cx - r, cx + r, cx + r, cx - r, cx - r],
                               [-cz - r, -cz - r, -cz + r, -cz + r, -cz - r])
                floor.set_3d_properties([ground] * 5)
            elif len(floor.get_xdata()) == 0:     # static floor, drawn once
                floor.set_data([gx0, gx1, gx1, gx0, gx0],
                               [-gz0, -gz0, -gz1, -gz1, -gz0])
                floor.set_3d_properties([ground] * 5)
            for ln, ch in zip(lines3d, CHAIN):
                vis = [j for j in ch if j not in missing]
                # Break the chain at the first missing joint: a segment drawn
                # across a gap would imply a bone that was never measured.
                if len(vis) < len(ch):
                    vis = []
                    for j in ch:
                        if j in missing:
                            break
                        vis.append(j)
                A = to_axes(P[vis])
                ln.set_data(A[:, 0], A[:, 1])
                ln.set_3d_properties(A[:, 2])
            if body_pc is not None:
                # The body goes through the same world -> axes map as the lines.
                Vb = to_axes(_so.fit_frame(P, body_model, missing=missing))
                Fb = body_model[1][body_keep]
                # Cull against the CURRENT camera: the panel can be dragged to
                # any angle, and a fixed view direction then hides the half of
                # the body that has turned towards the viewer.
                vis = _so.front_faces(Vb, Fb, _so.view_dir(ax3d.elev, ax3d.azim))
                tri = Vb[Fb[vis]]
                body_pc.set_verts(tri)
                # Lambert shade per face, otherwise a flat fill with no edges
                # is a featureless silhouette and the pose is unreadable.
                nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
                lam = np.clip(np.abs(nrm.dot(BODY_LIGHT)), 0, 1) ** 0.8
                body_pc.set_facecolor(np.clip(
                    BODY_RGB[None, :] * (0.42 + 0.58 * lam)[:, None], 0, 1))
        if playhead is not None:
            playhead.set_xdata([clip.emg_at(frame)] * 2)
        title.set_text(f"{clip.stem}    frame {frame + 1:>4}/{clip.n_frames}")
        return lines3d + ([playhead] if playhead is not None else [])

    draw(0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, draw


def save_video(clip, out_path, show_motion=True, show_emg=True,
               fps=FPS_DEFAULT, dpi=110, progress=None, follow=True,
               emg_scale='channel', show_body=False, body_detail=BODY_DETAIL,
               show_skeleton=True):
    """Render the clip to a video file. Returns the path written."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    fig, draw = build_figure(clip, show_motion, show_emg, dpi=dpi, follow=follow,
                             show_body=show_body, body_detail=body_detail,
                             show_skeleton=show_skeleton,
                             emg_scale=emg_scale)
    n = clip.n_frames

    def _frame(i):
        if progress and (i % 10 == 0 or i == n - 1):
            progress(i + 1, n)
        return draw(i)

    anim = animation.FuncAnimation(fig, _frame, frames=n, interval=1000 / fps,
                                   blit=False)
    ext = osp.splitext(out_path)[1].lower()
    if ext == '.gif':
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2400,
                                        metadata={'title': clip.stem})
    anim.save(out_path, writer=writer)
    plt.close(fig)
    return out_path


# ── GUI ──────────────────────────────────────────────────────────────────────



def _fit_axes_to_content(fig, ax, margin=0.02):
    """Grow a 3d axes so its drawn content fills the cell it was given.

    mplot3d sizes its axes from the data cube, not from what ends up visible,
    so a body inside a mostly-empty cube leaves wide margins -- and next to a
    second panel that reads as a gap between them. There is no API for the
    projected extent of the content, so this renders once, finds the bounding
    box of non-background pixels inside the axes, and rescales the axes rect
    about the point that keeps that content centred where it already is.
    """
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    H, W = buf.shape[:2]
    bb = ax.get_window_extent()
    x0, x1 = max(int(bb.x0), 0), min(int(np.ceil(bb.x1)), W)
    y0, y1 = max(int(bb.y0), 0), min(int(np.ceil(bb.y1)), H)
    if x1 <= x0 or y1 <= y0:
        return
    # buffer rows run top-down, window extents bottom-up
    sub = buf[H - y1:H - y0, x0:x1]
    ink = np.any(sub < 250, axis=2)
    if not ink.any():
        return
    rows, cols = np.where(ink)
    cw = (cols.max() - cols.min() + 1) / float(x1 - x0)
    ch = (rows.max() - rows.min() + 1) / float(y1 - y0)
    if cw <= 0 or ch <= 0:
        return
    s = min((1.0 - 2 * margin) / cw, (1.0 - 2 * margin) / ch)
    if s <= 1.001:
        return
    pos = ax.get_position()
    # content centre in figure fraction
    ccx = pos.x0 + pos.width * ((cols.min() + cols.max() + 1) / 2.0 / (x1 - x0))
    ccy = pos.y0 + pos.height * (1.0 - (rows.min() + rows.max() + 1) / 2.0 / (y1 - y0))
    dx, dy = pos.x0 + pos.width / 2.0, pos.y0 + pos.height / 2.0
    px = (dx - ccx * s) / (1.0 - s)
    py = (dy - ccy * s) / (1.0 - s)
    ax.set_position([px + (pos.x0 - px) * s, py + (pos.y0 - py) * s,
                     pos.width * s, pos.height * s])


def save_strip(clip, out_path, n_frames=10, dpi=200, show_body=True,
               body_detail=BODY_DETAIL, show_skeleton=False, figsize=None,
               show_emg=True, emg_scale='channel'):
    """Save one still of N equally spaced frames overlaid, last one opaque.

    The camera is fixed over the whole trajectory rather than following the
    root: with a following camera every frame is drawn at the same place and
    the overlay collapses into a single body. Fixed, the spacing between the
    bodies is the distance actually travelled.

    Returns the path written.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if clip.joints is None:
        raise ValueError('clip has no joints to draw')
    if not show_body and not show_skeleton:
        raise ValueError('nothing to draw: enable the SMPL body or the skeleton')
    n_frames = int(max(1, min(n_frames, clip.n_frames)))
    idx = np.linspace(0, clip.n_frames - 1, n_frames).round().astype(int)

    body_model = eye = None
    if show_body:
        import smpl_overlay as _so
        body_model = _so.load_model(detail=body_detail)
        eye = _so.view_dir(12, VIEW_AZIM)

    J = clip.joints
    lo, hi = J.reshape(-1, 3).min(0), J.reshape(-1, 3).max(0)
    stature = float(hi[1] - lo[1])
    ground = float(lo[1])
    # Box the actual path, not a forced square. Squaring the footprint keeps
    # the ground plane unsheared, but so does a box_aspect proportional to the
    # real ranges -- and squaring inflates the smaller axis to match the larger
    # (here 0.8 m of travel padded out to 2.5 m), leaving most of the panel as
    # empty floor and a wide gap before the EMG.
    pad = stature * 0.14
    x0, x1 = float(lo[0]) - pad, float(hi[0]) + pad
    z0, z1 = float(lo[2]) - pad, float(hi[2]) + pad
    xr, zr = x1 - x0, z1 - z0
    top = ground + stature * 1.08

    with_emg = bool(show_emg and clip.emg is not None)
    if figsize is None:
        figsize = (12.6, 5.4) if with_emg else (9.0, 5.4)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')
    if with_emg:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.02,
                              left=0.005, right=0.995, top=0.985, bottom=0.02)
        ax = fig.add_subplot(gs[0, 0], projection='3d')
        ax_emg = fig.add_subplot(gs[0, 1])
    else:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax_emg = None
    ax.set_xlim(x0, x1)
    ax.set_ylim(-z1, -z0)
    ax.set_zlim(ground, top)
    ax.set_box_aspect((xr, zr, top - ground))
    ax.set_axis_off()
    ax.view_init(elev=12, azim=VIEW_AZIM)
    ax.plot([x0, x1, x1, x0, x0], [-z0, -z0, -z1, -z1, -z0],
            [ground] * 5, color='#DCE3E6', lw=1.0, zorder=0)

    for k, f in enumerate(idx):
        last = (k == len(idx) - 1)
        P = J[f]
        if show_body:
            Vb = to_axes(_so.fit_frame(P, body_model))
            Fb = body_model[1]
            tri = Vb[Fb[_so.front_faces(Vb, Fb, eye)]]
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
            lam = np.clip(np.abs(nrm.dot(BODY_LIGHT)), 0, 1) ** 0.8
            # One grey throughout: the final frame is distinguished by being
            # opaque, not by changing colour.
            cols = np.clip(BODY_RGB[None, :] * (0.42 + 0.58 * lam)[:, None], 0, 1)
            alpha = 1.0 if last else 0.14 + 0.08 * (k / max(len(idx) - 2, 1))
            # Stroke the opaque frame's triangles in their own fill colour:
            # adjacent faces otherwise leave hairline antialiasing gaps that
            # show the background through and read as a visible triangulation.
            # Only safe where alpha is 1 -- on the ghosts it would double-blend.
            pc = Poly3DCollection(
                tri, facecolors=cols,
                edgecolors=cols if last else 'none',
                linewidths=0.3 if last else 0.0,
                alpha=alpha, zorder=2 + k)
            ax.add_collection3d(pc)
        if show_skeleton:
            a = 1.0 if last else 0.35
            for ch, side in zip(CHAIN, CHAIN_SIDE):
                A = to_axes(P[ch])
                ax.plot(A[:, 0], A[:, 1], A[:, 2], '-o', lw=2.2, ms=3.0,
                        color=COL[side], alpha=a, solid_capstyle='round',
                        zorder=20 + k)

    if ax_emg is not None:
        E = clip.emg
        live = np.abs(E).sum(1) > 0
        n = E.shape[1]
        t = np.arange(n)
        peaks = np.abs(E).max(axis=1)
        if emg_scale == 'muscle':
            pair = np.maximum(peaks[0::2], peaks[1::2])
            scale = np.repeat(pair, 2)[:E.shape[0]]
        else:
            scale = peaks
        scale = np.where(scale > 0, scale, 1.0)
        for i in range(E.shape[0]):
            y_off = E.shape[0] - 1 - i
            side = 'R' if EMG_LABELS[i][0] == 'R' else 'L'
            c = COL[side] if live[i] else '#C9D2D6'
            ax_emg.plot(t, y_off + 0.78 * np.abs(E[i]) / scale[i], lw=.75,
                        color=c)
            ax_emg.text(-n * 0.015, y_off, EMG_LABELS[i], ha='right',
                        va='center', fontsize=6.5, family='monospace',
                        color='#33424A' if live[i] else '#AEBCC2')
        # Mark where each rendered body sits in time, so posture can be read
        # against activation.
        for f in idx:
            ax_emg.axvline(clip.emg_at(int(f)), color='#C8811F', lw=0.6,
                           alpha=0.55, zorder=1)
        ax_emg.set_xlim(0, n - 1)
        ax_emg.set_ylim(-0.6, E.shape[0])
        ax_emg.set_yticks([])
        ax_emg.set_xticks([])
        for sp in ax_emg.spines.values():
            sp.set_visible(False)
    # tight_layout warns on mixed 2d/3d axes; the gridspec above already sets
    # the margins, so leave them alone.
    if not with_emg:
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    if ax_emg is not None:
        _fit_axes_to_content(fig, ax)
    # mplot3d reserves a large margin around its cube; trim it so the strip is
    # not mostly white space.
    fig.savefig(out_path, dpi=dpi, facecolor='white', bbox_inches='tight',
                pad_inches=0.12)
    plt.close(fig)
    return out_path


def run_gui(clips, rel, source, fps, follow=True, emg_scale='channel',
            show_body=False, body_detail=BODY_DETAIL, show_skeleton=True):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt

    win = tk.Tk()
    win.title('MoE3D clip viewer')
    win.geometry('1180x680')

    state = {'clips': list(clips), 'i': -1, 'frame': 0, 'playing': False,
             'canvas': None, 'draw': None, 'fig': None, 'job': None,
             'pick_dir': rel.pick_dir()}

    left = ttk.Frame(win, padding=8); left.pack(side='left', fill='y')
    right = ttk.Frame(win, padding=(0, 8, 8, 8)); right.pack(side='right', fill='both', expand=True)

    ttk.Label(left, text='Clips', font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
    lb = tk.Listbox(left, width=38, height=26, exportselection=False,
                    activestyle='none')
    lb.pack(fill='y', expand=True, pady=(4, 6))

    btns = ttk.Frame(left); btns.pack(fill='x')
    show_motion = tk.BooleanVar(value=True)
    show_emg = tk.BooleanVar(value=True)
    follow_var = tk.BooleanVar(value=follow)
    pair_var = tk.BooleanVar(value=(emg_scale == 'muscle'))
    body_var = tk.BooleanVar(value=show_body)
    skel_var = tk.BooleanVar(value=show_skeleton)
    strip_n = tk.IntVar(value=10)
    src_var = tk.StringVar(value=source)
    status = tk.StringVar(value='Add clips to begin.')

    def refresh_list():
        lb.delete(0, 'end')
        for c in state['clips']:
            lb.insert('end', c.stem)

    def add_files():
        paths = filedialog.askopenfilenames(
            title='Select clips',
            initialdir=state['pick_dir'],
            filetypes=[('Clip data', '*.npy *.npz'), ('All files', '*.*')])
        if paths:
            state['pick_dir'] = osp.dirname(paths[0])
        added = 0
        for p in paths:
            try:
                state['clips'].append(load_clip(p, rel, src_var.get()))
                added += 1
            except Exception as exc:
                messagebox.showwarning('Could not load', f'{osp.basename(p)}\n\n{exc}')
        if added:
            refresh_list()
            lb.selection_clear(0, 'end')
            lb.selection_set(len(state['clips']) - added)
            select()

    def remove_sel():
        s = lb.curselection()
        if not s:
            return
        del state['clips'][s[0]]
        state['i'] = -1
        refresh_list()
        status.set('Removed.')

    ttk.Button(btns, text='Add files…', command=add_files).pack(side='left')
    ttk.Button(btns, text='Remove', command=remove_sel).pack(side='left', padx=6)

    opts = ttk.LabelFrame(left, text='Show', padding=6)
    opts.pack(fill='x', pady=(10, 0))

    def rebuild(*_):
        if state['i'] >= 0:
            render(state['i'], keep_frame=True)

    ttk.Checkbutton(opts, text='Motion', variable=show_motion,
                    command=rebuild).pack(anchor='w')
    ttk.Checkbutton(opts, text='EMG', variable=show_emg,
                    command=rebuild).pack(anchor='w')
    ttk.Checkbutton(opts, text='Scale L/R together', variable=pair_var,
                    command=rebuild).pack(anchor='w', padx=(16, 0))
    ttk.Label(opts, text='off = each trace to its own peak\non  = each L/R pair shares a\n     peak, so sides compare',
              foreground='#66757B', font=('TkDefaultFont', 8)).pack(anchor='w', padx=(16, 0))
    ttk.Separator(opts, orient='horizontal').pack(fill='x', pady=5)
    ttk.Checkbutton(opts, text='Follow subject', variable=follow_var,
                    command=rebuild).pack(anchor='w')
    ttk.Separator(opts, orient='horizontal').pack(fill='x', pady=5)
    ttk.Checkbutton(opts, text='Skeleton', variable=skel_var,
                    command=rebuild).pack(anchor='w')
    ttk.Checkbutton(opts, text='SMPL body', variable=body_var,
                    command=rebuild).pack(anchor='w')
    ttk.Label(opts, text='neutral mesh fitted to the joints\nplayback is slower while on'
                         '\nneeds the SMPL-H model (README)',
              foreground='#66757B', font=('TkDefaultFont', 8)).pack(anchor='w', padx=(16, 0))

    def switch_source():
        """Reload every clip from the selected joints source.

        Switching between 263-dim and SMPLH_AMASS maps clips to their trial
        and a trial to all of its clips.
        """
        want = src_var.get()
        kept, missing, seen = [], [], set()
        for c in state['clips']:
            stems = resolve_stems(c.stem, rel, want)
            if not stems and len(c.stem) > 2:           # trial -> its clips
                stems = resolve_stems(c.stem + '0', rel, want) or \
                    resolve_stems(c.stem, rel, want)
            for st in stems or [c.stem]:
                if st in seen:
                    continue
                seen.add(st)
                try:
                    kept.append(load_clip(st, rel, want))
                except Exception:
                    missing.append(st)
        if not kept and missing:
            messagebox.showwarning(
                'Not available',
                'No loaded clip exists in %s.' % SOURCE_LABELS[want])
            return
        state['clips'] = kept
        refresh_list()
        if kept:
            i = min(max(state['i'], 0), len(kept) - 1)
            lb.selection_clear(0, 'end'); lb.selection_set(i)
            render(i, keep_frame=True)
        if missing:
            status.set('%d clip(s) absent from that source, dropped.' % len(missing))

    ttk.Separator(opts, orient='horizontal').pack(fill='x', pady=5)
    ttk.Label(opts, text='Joints source').pack(anchor='w')
    for key in SOURCES:
        ttk.Radiobutton(opts, text=SOURCE_LABELS[key], value=key,
                        variable=src_var, command=switch_source).pack(anchor='w', padx=(10, 0))
    ttk.Label(opts, text='translated = treadmill belt\n     cancelled; overground\n     clips are unchanged',
              foreground='#66757B', font=('TkDefaultFont', 8)).pack(anchor='w', padx=(10, 0))

    def do_save():
        if state['i'] < 0:
            messagebox.showinfo('Nothing to save', 'Select a clip first.')
            return
        clip = state['clips'][state['i']]
        out = filedialog.asksaveasfilename(
            title='Save video as', defaultextension='.mp4',
            initialfile=clip.stem + '.mp4',
            filetypes=[('MP4 video', '*.mp4'), ('Animated GIF', '*.gif')])
        if not out:
            return
        stop()
        status.set('Rendering…'); win.update_idletasks()
        try:
            save_video(clip, out, show_motion.get(), show_emg.get(), fps=fps,
                       show_body=body_var.get(), body_detail=body_detail,
                       show_skeleton=skel_var.get(),
                       follow=follow_var.get(),
                       emg_scale='muscle' if pair_var.get() else 'channel',
                       progress=lambda i, n: (status.set(f'Rendering frame {i}/{n}…'),
                                              win.update_idletasks()))
            status.set(f'Saved  {out}')
        except Exception as exc:
            status.set('Save failed.')
            messagebox.showerror('Save failed', str(exc))
        finally:
            render(state['i'], keep_frame=True)

    def do_save_figure():
        """Overlay N equally spaced frames into one still."""
        if state['i'] < 0:
            messagebox.showinfo('Nothing to save', 'Select a clip first.')
            return
        clip = state['clips'][state['i']]
        try:
            n = int(strip_n.get())
        except Exception:
            messagebox.showerror('Bad frame count', 'Enter a whole number.')
            return
        if n < 1:
            messagebox.showerror('Bad frame count', 'Need at least 1 frame.')
            return
        out = filedialog.asksaveasfilename(
            title='Save figure as', defaultextension='.png',
            initialfile=f'{clip.stem}_{n}frames.png',
            filetypes=[('PNG image', '*.png'), ('PDF', '*.pdf'),
                       ('SVG', '*.svg')])
        if not out:
            return
        stop()
        status.set('Rendering figure…'); win.update_idletasks()
        try:
            save_strip(clip, out, n_frames=n, show_body=body_var.get(),
                       body_detail=body_detail, show_skeleton=skel_var.get())
            status.set(f'Saved  {out}')
        except Exception as exc:
            status.set('Save failed.')
            messagebox.showerror('Save failed', str(exc))
        finally:
            render(state['i'], keep_frame=True)

    ttk.Button(left, text='Save video…', command=do_save).pack(fill='x', pady=(10, 0))

    figrow = ttk.Frame(left)
    figrow.pack(fill='x', pady=(6, 0))
    ttk.Button(figrow, text='Save figure…',
               command=do_save_figure).pack(side='left', fill='x', expand=True)
    ttk.Label(figrow, text='frames').pack(side='left', padx=(6, 2))
    ttk.Spinbox(figrow, from_=1, to=60, width=4,
                textvariable=strip_n).pack(side='left')

    # transport
    bar = ttk.Frame(right); bar.pack(fill='x', side='bottom', pady=(6, 0))
    play_btn = ttk.Button(bar, text='Play', width=7)
    play_btn.pack(side='left')
    scrub = ttk.Scale(bar, from_=0, to=1, orient='horizontal')
    scrub.pack(side='left', fill='x', expand=True, padx=8)
    ttk.Label(bar, textvariable=status).pack(side='bottom', anchor='w')

    holder = ttk.Frame(right); holder.pack(fill='both', expand=True)

    def stop():
        state['playing'] = False
        play_btn.config(text='Play')
        if state['job']:
            win.after_cancel(state['job']); state['job'] = None

    def render(i, keep_frame=False):
        stop()
        clip = state['clips'][i]
        state['i'] = i
        if not keep_frame:
            state['frame'] = 0
        if state['canvas'] is not None:
            state['canvas'].get_tk_widget().destroy()
            plt.close(state['fig'])
        try:
            fig, draw = build_figure(clip, show_motion.get(), show_emg.get(),
                                     follow=follow_var.get(),
                                     emg_scale='muscle' if pair_var.get() else 'channel',
                                     show_body=body_var.get(),
                                     body_detail=body_detail,
                                     show_skeleton=skel_var.get())
        except ValueError as exc:
            status.set(str(exc)); state['canvas'] = None; return
        cv = FigureCanvasTkAgg(fig, master=holder)
        cv.get_tk_widget().pack(fill='both', expand=True)
        state.update(canvas=cv, draw=draw, fig=fig)

        def on_drag(event):
            # Rotating the 3-D panel changes which faces of the body point
            # at the camera; when paused, nothing else would redraw them.
            if (event.button and not state['playing']
                    and getattr(event.inaxes, 'name', '') == '3d'):
                draw(state['frame']); cv.draw_idle()
        cv.mpl_connect('motion_notify_event', on_drag)
        cv.mpl_connect('button_release_event', on_drag)
        scrub.config(to=max(1, clip.n_frames - 1))
        state['frame'] = min(state['frame'], clip.n_frames - 1)
        scrub.set(state['frame'])
        draw(state['frame']); cv.draw()
        status.set(f'{clip.stem}   {clip.describe()}')

    def select(*_):
        s = lb.curselection()
        if s:
            render(s[0])

    lb.bind('<<ListboxSelect>>', select)

    def tick():
        if not state['playing'] or state['draw'] is None:
            return
        clip = state['clips'][state['i']]
        state['frame'] = (state['frame'] + 1) % clip.n_frames
        scrub.set(state['frame'])
        state['draw'](state['frame']); state['canvas'].draw_idle()
        state['job'] = win.after(int(1000 / fps), tick)

    def toggle():
        if state['i'] < 0:
            return
        if state['playing']:
            stop()
        else:
            state['playing'] = True
            play_btn.config(text='Pause')
            tick()

    play_btn.config(command=toggle)

    def on_scrub(v):
        if state['draw'] is None or state['playing']:
            return
        state['frame'] = int(float(v))
        state['draw'](state['frame']); state['canvas'].draw_idle()

    scrub.config(command=on_scrub)
    win.bind('<space>', lambda e: toggle())

    refresh_list()
    if state['clips']:
        lb.selection_set(0); render(0)
    win.mainloop()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    here = osp.dirname(osp.dirname(osp.abspath(__file__)))
    ap = argparse.ArgumentParser(
        description='Play and export MoE3D clips (skeleton + EMG).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage\n-----\n')[-1])
    ap.add_argument('clips', nargs='*', help='clip names, trial prefixes or .npy paths')
    ap.add_argument('--data-root', default=here,
                    help='the MoE3D folder (default: the parent of this script)')
    ap.add_argument('--source', choices=list(SOURCES), default='hml3d',
                    help='which joints to read: hml3d (HumanML3D skeleton, '
                         'default), subject (each subject\'s own skeleton), '
                         'amass (SMPL-H trials), or the -tt variants with '
                         'treadmill trials translated')
    ap.add_argument('--save', metavar='OUT',
                    help='render to video and exit (no GUI); .mp4 or .gif')
    ap.add_argument('--out-dir', metavar='DIR',
                    help='with multiple clips and --save, write OUT-named files here')
    ap.add_argument('--global-frame', action='store_true',
                    help='fix the camera over the whole path so the subject is\n'
                         'seen travelling, instead of keeping it centred')
    ap.add_argument('--emg-scale', choices=['channel', 'muscle'], default='channel',
                    help="'channel' scales each trace to its own peak; 'muscle'\n"
                         'scales each L/R pair together so the two sides are\n'
                         'visually comparable')
    ap.add_argument('--body', action='store_true',
                    help='overlay the neutral SMPL body fitted to the joints')
    ap.add_argument('--smplh-model', metavar='NPZ',
                    help='SMPL-H neutral model.npz, for --body and for exact '
                         'SMPLH_AMASS skeletons; male/ and female/ beside '
                         'neutral/ are used too (default: '
                         'visualisation/body_models/smplh/neutral/model.npz)')
    ap.add_argument('--body-detail', type=float, default=BODY_DETAIL,
                    metavar='M',
                    help='mesh decimation cell in metres; smaller = finer and '
                         'slower (default %(default)s)')
    ap.add_argument('--no-skeleton', action='store_true',
                    help='hide the coloured limb skeleton (use with --body)')
    ap.add_argument('--save-figure', metavar='OUT',
                    help='write a still of overlapping frames instead of a '
                         'video; all translucent except the last')
    ap.add_argument('--figure-frames', type=int, default=10, metavar='N',
                    help='frames to overlay for --save-figure '
                         '(default %(default)s)')
    ap.add_argument('--no-motion', action='store_true')
    ap.add_argument('--no-emg', action='store_true')
    ap.add_argument('--fps', type=float, default=FPS_DEFAULT)
    ap.add_argument('--dpi', type=int, default=110)
    args = ap.parse_args()

    import smpl_overlay
    if args.smplh_model:
        smpl_overlay.MODEL = osp.abspath(args.smplh_model)
    if args.body and not osp.isfile(smpl_overlay.MODEL):
        sys.exit(f"[ERROR] --body needs the SMPL-H model, not found at "
                 f"{smpl_overlay.MODEL}.\n  Download SMPL-H from mano.is.tue.mpg.de "
                 f"and put neutral/model.npz there, or pass --smplh-model.")

    rel = Release(osp.abspath(args.data_root))
    if not rel.ok():
        print(f"[warning] no data under {rel.motion}; only files inside "
              f"dataset_build/ can be opened (or pass --data-root)", file=sys.stderr)

    tokens = []
    for t in args.clips:
        tokens.extend(sorted(glob(t)) if any(ch in t for ch in '*?[') else [t])

    clips = []
    for t in tokens:
        stems = resolve_stems(t, rel, args.source)
        if not stems:
            print(f"[skip] {t}: no clip matches that stem or prefix", file=sys.stderr)
            continue
        if len(stems) > 1:
            print(f"{t} -> {len(stems)} slices", file=sys.stderr)
        for st in stems:
            try:
                clips.append(load_clip(st, rel, args.source))
            except Exception as exc:
                print(f"[skip] {st}: {exc}", file=sys.stderr)

    if args.save_figure:
        if not clips:
            sys.exit('[ERROR] --save-figure needs at least one clip')
        for c in clips:
            out = args.save_figure if len(clips) == 1 else osp.join(
                args.out_dir or osp.dirname(args.save_figure) or '.',
                c.stem + osp.splitext(args.save_figure)[1])
            os.makedirs(osp.dirname(osp.abspath(out)), exist_ok=True)
            try:
                save_strip(c, out, n_frames=args.figure_frames,
                           dpi=args.dpi, show_body=args.body,
                           body_detail=args.body_detail,
                           show_skeleton=not args.no_skeleton)
            except ValueError as exc:
                sys.exit(f'[ERROR] {exc}')
            print(f'wrote {out}')
        return 0

    if args.save:
        if not clips:
            sys.exit('[ERROR] --save needs at least one clip')
        outs = []
        for c in clips:
            out = args.save if len(clips) == 1 else osp.join(
                args.out_dir or osp.dirname(args.save) or '.',
                c.stem + osp.splitext(args.save)[1])
            os.makedirs(osp.dirname(osp.abspath(out)), exist_ok=True)
            tty = sys.stderr.isatty()
            save_video(c, out, not args.no_motion, not args.no_emg,
                       fps=args.fps, dpi=args.dpi, follow=not args.global_frame,
                       emg_scale=args.emg_scale, show_body=args.body,
                       body_detail=args.body_detail,
                       show_skeleton=not args.no_skeleton,
                       progress=(lambda i, n: print(f"\r  {c.stem}  {i}/{n}",
                                                    end='', file=sys.stderr))
                       if tty else None)
            print(("\r" if tty else "") + f"wrote {out}")
            outs.append(out)
        return 0

    run_gui(clips, rel, args.source, args.fps,
            follow=not args.global_frame, emg_scale=args.emg_scale,
            show_body=args.body, body_detail=args.body_detail,
            show_skeleton=not args.no_skeleton)
    return 0


if __name__ == '__main__':
    sys.exit(main())
