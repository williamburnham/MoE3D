#!/usr/bin/env python
"""Pose the neutral SMPL-H body onto a frame of 22 HumanML3D joints.

The clips carry joint positions, not SMPL parameters, so the body cannot simply
be replayed -- it has to be fitted. Rather than run an optimiser per frame
(far too slow for playback) this solves each joint's global rotation in closed
form with Kabsch: for every joint, align the rest-pose offsets of its children
to the offsets the data actually shows, then skin the template with those
transforms.

Two consequences worth knowing:

* Twist about a bone's own axis is unobservable from joint positions alone, so
  forearm and shank twist are not recovered. Gait reads correctly; a wrist
  rotation does not.
* The mesh keeps the template's bone lengths. Fitted clips are retargeted onto
  the HumanML3D template by uniform_skeleton() so they match closely, but raw
  clips preserve each subject's own limb lengths and the body will not track
  them exactly. Pass scale=True to apply a single global scale from overall
  stature, which absorbs most of the difference.
"""
import os.path as osp

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
# The SMPL-H model is licensed separately (mano.is.tue.mpg.de) and not
# distributed with MoE3D. Put its neutral model.npz here, or pass
# visualise_tool.py --smplh-model.
MODEL = osp.join(HERE, 'body_models', 'smplh', 'neutral', 'model.npz')

N_BODY = 22
_CACHE = {}


def decimate(V, F, W, cell):
    """Vertex-clustering decimation: merge vertices sharing a grid cell.

    Interactive playback cannot afford 13,776 triangles -- matplotlib takes
    ~460 ms to draw that many. Dropping every Nth face would be cheaper but
    punches holes in the surface; clustering keeps a closed shell. Skinning
    weights are averaged over each cluster so the reduced mesh still deforms.
    """
    key = np.floor(V / cell).astype(np.int64)
    _u, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)      # numpy 2.0.0 returns it as an (n, 1) column
    n = inv.max() + 1
    Vd = np.zeros((n, 3))
    Wd = np.zeros((n, W.shape[1]))
    cnt = np.bincount(inv, minlength=n).astype(float)[:, None]
    np.add.at(Vd, inv, V)
    np.add.at(Wd, inv, W)
    Vd /= cnt
    Wd /= cnt
    Fd = inv[F]
    ok = ((Fd[:, 0] != Fd[:, 1]) & (Fd[:, 1] != Fd[:, 2]) & (Fd[:, 0] != Fd[:, 2]))
    Fd = Fd[ok]
    # Deduplicate on a CYCLIC key, not a sorted one. (a,b,c) and (a,c,b) share
    # a vertex set but are opposite-wound faces -- in thin regions, where the
    # near and far surfaces fall into the same clusters, both are real and
    # dropping one punches a hole in the shell. Rolling each face so its
    # smallest index leads keeps rotations equal while keeping reflections
    # distinct.
    roll = np.argmin(Fd, axis=1)
    rows = np.arange(len(Fd))[:, None]
    cols = (roll[:, None] + np.arange(3)[None, :]) % 3
    key = Fd[rows, cols]
    _k, idx = np.unique(key, axis=0, return_index=True)
    Fd = Fd[np.sort(idx)]
    return Vd, Fd, Wd


def load_model(path=None, detail=None):
    """(v_template, faces, rest_joints, weights, parents) for the neutral body.

    ``detail`` is a clustering cell size in metres; None keeps the full mesh.
    """
    key = (path or MODEL, detail)
    if key in _CACHE:
        return _CACHE[key]
    full = _load_full(path)
    if detail is None:
        _CACHE[key] = full
        return full
    V, F, J, W, parents = full
    Vd, Fd, Wd = decimate(V, F, W, detail)
    _CACHE[key] = (Vd, Fd, J, Wd, parents)
    return _CACHE[key]


def _load_full(path=None):
    key = ('__full__', path or MODEL)
    if key in _CACHE:
        return _CACHE[key]
    if not osp.isfile(path or MODEL):
        raise FileNotFoundError(
            'SMPL-H model not found at %s; download SMPL-H from '
            'mano.is.tue.mpg.de and put neutral/model.npz there, or pass '
            '--smplh-model' % (path or MODEL))
    d = np.load(path or MODEL, allow_pickle=True)
    V = np.asarray(d['v_template'], np.float64)
    F = np.asarray(d['f'], np.int64)
    J = np.asarray(d['J_regressor'], np.float64).dot(V)
    W = np.asarray(d['weights'], np.float64)
    parents = np.asarray(d['kintree_table'], np.int64)[0].copy()
    parents[0] = -1
    _CACHE[key] = (V, F, J, W, parents)
    return _CACHE[key]


def _kabsch(A, B):
    """Rotation R minimising |R A - B| for paired 3-vector sets (no scaling)."""
    H = A.T.dot(B)
    U, _s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T.dot(U.T)))
    D = np.diag([1.0, 1.0, d])
    return Vt.T.dot(D).dot(U.T)


def global_rotations(P, J0, parents, n_body=N_BODY, missing=()):
    """Per-joint global rotations aligning the rest skeleton to P [n_body,3].

    `missing` names joints stored as zeros because the cohort never measured
    them. They are excluded from the alignment and inherit their parent's
    rotation, so the body keeps a plausible rest posture there instead of being
    dragged toward the world origin.
    """
    children = {j: [] for j in range(len(parents))}
    for j, p in enumerate(parents):
        if p >= 0:
            children[p].append(j)

    R = [np.eye(3) for _ in range(len(parents))]
    order = sorted(range(len(parents)), key=lambda j: _depth(j, parents))
    for j in order:
        kids = [c for c in children[j] if c < n_body and c not in missing]
        if j in missing:
            kids = []
        if kids:
            A = np.array([J0[c] - J0[j] for c in kids])
            B = np.array([P[c] - P[j] for c in kids])
            if len(kids) == 1:
                R[j] = _align_vectors(A[0], B[0])
            else:
                R[j] = _kabsch(A, B)
        elif parents[j] >= 0:
            R[j] = R[parents[j]]            # leaf / hand: inherit the parent
    return R


def _depth(j, parents):
    d = 0
    while parents[j] >= 0:
        j = parents[j]
        d += 1
    return d


def _align_vectors(a, b):
    """Minimal rotation taking direction a to direction b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(a.dot(b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K.dot(K) * (1.0 / (1.0 + c))


def fit_frame(P, model=None, scale=True, missing=()):
    """Vertices [6890,3] of the neutral body posed onto joints P [>=22,3]."""
    V, F, J0, W, parents = model or load_model()
    P = np.asarray(P, np.float64)[:N_BODY]

    s = 1.0
    if scale:
        # One global scale from total bone length: the template is a fixed size
        # and raw clips keep each subject's own limb lengths.
        li = [(j, parents[j]) for j in range(1, N_BODY)
              if j not in missing and parents[j] not in missing]
        num = sum(np.linalg.norm(P[j] - P[p]) for j, p in li)
        den = sum(np.linalg.norm(J0[j] - J0[p]) for j, p in li)
        if den > 1e-9:
            s = float(num / den)

    R = global_rotations(P, J0 * s, parents, missing=set(missing))
    Rm = np.stack(R)                                   # [nJ,3,3]
    t = P[:N_BODY] - np.einsum('jab,jb->ja', Rm[:N_BODY], J0[:N_BODY] * s)
    # A missing joint has no measured position to translate to; place it by
    # forward kinematics from its parent instead.
    for j in range(1, N_BODY):
        if j in set(missing):
            p = parents[j]
            t[j] = t[p] if p >= 0 else t[j]
    T = np.zeros((len(parents), 3))
    T[:N_BODY] = t
    for j in range(N_BODY, len(parents)):              # hands follow the wrist
        p = parents[j]
        T[j] = T[p] if p >= 0 else 0.0

    # Blend the per-joint rotations into one 3x3 per vertex first. The direct
    # einsum over (vertices, joints, 3, 3) is ~15x slower and dominates the
    # frame budget during playback.
    Vs = V * s
    Tv = np.tensordot(W, Rm, axes=([1], [0]))          # [V,3,3]
    out = np.einsum('vab,vb->va', Tv, Vs) + W.dot(T)
    return out


def faces(model=None):
    return (model or load_model())[1]


def visible_faces_mask(model, missing):
    """Faces to keep when some joints were never measured.

    A vertex is dropped when most of its skinning weight belongs to missing
    joints, and a face when any of its vertices is dropped. Without this the
    overlay renders a rest-pose torso and T-pose arms over a cohort that
    measured neither, which looks like data rather than like absence.
    """
    V, F, J, W, parents = model
    if not missing:
        return np.ones(len(F), bool)
    miss = np.zeros(W.shape[1], bool)
    for j in missing:
        miss[j] = True
    # hand joints follow their wrist
    for j in range(N_BODY, W.shape[1]):
        p = parents[j]
        while p >= N_BODY:
            p = parents[p]
        if miss[p]:
            miss[j] = True
    w_missing = W[:, miss].sum(axis=1)
    drop_v = w_missing > 0.5
    return ~drop_v[F].any(axis=1)


def view_dir(elev_deg, azim_deg):
    """Unit vector toward the camera, in matplotlib-3d axes coordinates."""
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def front_faces(Vax, F, eye):
    """Indices of faces pointing at the camera, painted back to front.

    Halves the triangles matplotlib has to draw, and removes the far surface
    showing through the near one -- mplot3d has no depth buffer, so without
    this the interior of the body is visible.
    """
    tri = Vax[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    keep = np.where(n.dot(eye) > 0)[0]
    return keep[np.argsort(tri[keep].mean(axis=1).dot(eye))]
