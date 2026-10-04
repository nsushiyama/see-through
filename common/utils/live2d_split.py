"""Live2D / Spine oriented detailed split for See-through outputs.

Pure post-processing (numpy + OpenCV, **no torch**) that runs AFTER the normal See-through
pipeline (``apply_layerdiff`` -> ``apply_marigold``) on the inpainted semantic RGBA layers
it wrote to ``<save_dir>/<srcname>/<tag>.png``.  Nothing here modifies those files or the
normal ``further_extr`` path.

Processing order (fixed, see AGENT/MASTER_GOAL.md):
  semantic RGBA (full canvas, hidden parts inpainted) -> LR split (same rule as the existing
  ``part_lr_split``) -> split masks (connected components / shape analysis / optional external
  mask providers) -> intersect semantic RGBA with masks -> Live2D parts -> joint overlap ->
  (export to original canvas, visible-pixel restore) -> PSD.

Conventions
  * Every part is a FULL-CANVAS RGBA array (never cropped); PSD layers use offset (0, 0).
  * Side suffix ``_L`` / ``_R`` = CHARACTER's own left / right (front-facing character:
    character-right is on the SCREEN LEFT).  This matches the existing ``part_lr_split``
    which names the screen-left component ``<tag>-r``.
  * depth: larger = farther (See-through convention).
"""
from __future__ import annotations

import json
import logging
import os.path as osp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger('live2d_split')

ALPHA_T = 10  # same threshold as io_utils.load_part / save_part

V3_TAGS = ['front hair', 'back hair', 'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear',
           'footwear', 'tail', 'wings', 'objects', 'headwear', 'face', 'irides', 'eyebrow', 'eyewhite',
           'eyelash', 'eyewear', 'ears', 'earwear', 'nose', 'mouth']
V2_TAGS = ['hair', 'headwear', 'face', 'eyes', 'eyewear', 'ears', 'earwear', 'nose', 'mouth',
           'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear',
           'tail', 'wings', 'objects']

# tags the existing further_extr(tblr_split=True) splits into left/right
EXISTING_LR_TAGS = ['handwear', 'eyewhite', 'irides', 'eyelash', 'eyebrow', 'ears']

GROUP_OF_TAG = {
    'front hair': 'Hair', 'back hair': 'Hair', 'hair': 'Hair',
    'face': 'Face', 'ears': 'Face', 'eyebrow': 'Face', 'eyewhite': 'Face', 'irides': 'Face', 'eyelash': 'Face',
    'eyes': 'Face', 'nose': 'Face', 'mouth': 'Face', 'eyewear': 'Accessory', 'earwear': 'Accessory',
    'neck': 'Body', 'handwear': 'Body', 'legwear': 'Body', 'footwear': 'Body',
    'topwear': 'Clothes', 'bottomwear': 'Clothes',
    'headwear': 'Accessory', 'neckwear': 'Accessory', 'objects': 'Accessory', 'tail': 'Accessory', 'wings': 'Accessory',
}
# base layer name (before side suffix) used in the PSD
BASE_NAME = {'front hair': 'front_hair', 'back hair': 'back_hair', 'ears': 'ear', 'eyewhite': 'eye_white',
             'irides': 'iris', 'eyebrow': 'eyebrow', 'eyelash': 'eyelash', 'handwear': 'arm',
             'legwear': 'leg', 'footwear': 'foot'}
GROUP_ORDER = ['Hair', 'Face', 'Body', 'Clothes', 'Accessory', 'Other']


def base_name(tag: str) -> str:
    return BASE_NAME.get(tag, tag.replace(' ', '_'))


@dataclass
class Part:
    name: str
    img: np.ndarray            # (H, W, 4) uint8, full canvas
    depth: float               # depth median, larger = farther
    source: str                # semantic tag this part came from
    group: str = 'Other'
    side: Optional[str] = None  # 'character_left' / 'character_right' / None
    method: str = 'semantic'   # how it was produced
    depth_map: Optional[np.ndarray] = None

    @property
    def alpha(self) -> np.ndarray:
        return self.img[..., 3]

    @property
    def mask(self) -> np.ndarray:
        return self.img[..., 3] > ALPHA_T

    def area(self) -> int:
        return int(self.mask.sum())

    def derive(self, name, keep: np.ndarray, side=None, method=None, group=None) -> 'Part':
        """New part = this part's RGBA with alpha multiplied by ``keep`` (bool or float 0..1)."""
        img = self.img.copy()
        if keep.dtype == bool:
            img[..., 3] = np.where(keep, img[..., 3], 0).astype(np.uint8)
        else:
            img[..., 3] = np.round(img[..., 3].astype(np.float32) * np.clip(keep, 0, 1)).astype(np.uint8)
        img[img[..., 3] == 0, :3] = 0
        return Part(name, img, self.depth, self.source, group or self.group, side, method or self.method)


@dataclass
class SplitReport:
    stages: List[Tuple[str, int]] = field(default_factory=list)
    events: List[str] = field(default_factory=list)   # success / fallback messages

    def stage(self, label, n):
        self.stages.append((label, n))
        logger.info(f'{label}: {n}')
        print(f'[live2d] {label}: {n}', flush=True)

    def event(self, msg):
        self.events.append(msg)
        logger.info(msg)
        print(f'[live2d] {msg}', flush=True)


# ----------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------

def load_semantic_layers(srcd: str, tags: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, List[Part]]:
    """Load See-through semantic RGBA layers as full-canvas Parts.

    Unlike ``io_utils.load_part`` this keeps every non-empty layer (no bottom/right 10% strip
    rule) and never crops.  Returns (fullpage RGBA of src_img.png, parts sorted far -> near).
    """
    fullpage = np.array(Image.open(osp.join(srcd, 'src_img.png')).convert('RGBA'))
    H, W = fullpage.shape[:2]
    if tags is None:
        infop = osp.join(srcd, 'info.json')
        if osp.exists(infop):
            tags = list(json.load(open(infop))['parts'].keys())
        else:
            tags = [t for t in V3_TAGS + V2_TAGS if osp.exists(osp.join(srcd, t + '.png'))]
    parts = []
    seen = set()
    for tag in tags:
        p = osp.join(srcd, tag + '.png')
        if tag in seen or not osp.exists(p):
            continue
        seen.add(tag)
        img = np.array(Image.open(p).convert('RGBA'))
        if img.shape[:2] != (H, W):
            raise ValueError(f'{tag}.png has size {img.shape[:2]}, expected {(H, W)}')
        m = img[..., 3] > ALPHA_T
        if not m.any():
            continue
        dp = osp.join(srcd, tag + '_depth.png')
        if osp.exists(dp):
            dmap = np.array(Image.open(dp).convert('L')).astype(np.float32) / 255.
            depth = float(np.median(dmap[m]))
        else:
            dmap, depth = None, 1.0
        parts.append(Part(name=base_name(tag), img=img, depth=depth, source=tag,
                          group=GROUP_OF_TAG.get(tag, 'Other'), depth_map=dmap))
    parts.sort(key=lambda x: x.depth, reverse=True)
    return fullpage, parts


# ----------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------

def long_side(shape) -> int:
    return int(max(shape[:2]))


def overlap_px(shape, ratio=0.02) -> int:
    """Joint overlap in pixels: ``ratio`` of the long side, ratio clamped to [1%, 3%]."""
    ratio = float(np.clip(ratio, 0.01, 0.03))
    return max(2, int(round(long_side(shape) * ratio)))


def components(mask: np.ndarray, min_area: int = 1):
    """Connected components (8-conn) sorted by area desc: list of (bool mask, stats, centroid)."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out.append((lab == i, stats[i], cent[i]))
    out.sort(key=lambda c: -c[1][cv2.CC_STAT_AREA])
    return out


def assign_to_nearest(mask: np.ndarray, seeds: List[np.ndarray]) -> List[np.ndarray]:
    """Partition ``mask`` among ``seeds`` (disjoint bool masks inside it) by nearest seed pixel."""
    dists = []
    for s in seeds:
        inv = (~s).astype(np.uint8)
        dists.append(cv2.distanceTransform(inv, cv2.DIST_L2, 3))
    idx = np.argmin(np.stack(dists, 0), 0)
    return [mask & (idx == i) for i in range(len(seeds))]


def bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ----------------------------------------------------------------------------------------
# validation / fallback
# ----------------------------------------------------------------------------------------

def validate_split(src: Part, outs: List[Part], min_cover=0.995, min_part_area=16) -> Optional[str]:
    """Return None if ``outs`` is an acceptable replacement for ``src``, else a reason string."""
    if len(outs) < 2:
        return 'produced < 2 parts'
    sm = src.mask
    union = np.zeros_like(sm)
    for o in outs:
        if o.img.shape != src.img.shape:
            return f'{o.name}: canvas {o.img.shape} != {src.img.shape}'
        a = o.area()
        if a < min_part_area:
            return f'{o.name}: too small ({a}px)'
        if np.any(o.mask & ~sm):
            return f'{o.name}: pixels outside source alpha'
        union |= o.mask
    cover = union[sm].mean() if sm.any() else 1.0
    if cover < min_cover:
        return f'coverage {cover:.4f} < {min_cover}'
    return None


def safe_split(src: Part, fn: Callable[[Part], List[Part]], report: SplitReport, label: str, **vkw) -> List[Part]:
    """Run splitter ``fn``; on exception or failed validation keep the ORIGINAL layer."""
    try:
        outs = fn(src)
    except Exception as e:  # noqa: BLE001 - any failure must fall back
        report.event(f'fallback {label} ({src.name}): exception {type(e).__name__}: {e}')
        return [src]
    if outs is None or (len(outs) == 1 and outs[0] is src):
        report.event(f'keep {label} ({src.name}): nothing to split')
        return [src]
    reason = validate_split(src, outs, **vkw)
    if reason is not None:
        report.event(f'fallback {label} ({src.name}): {reason}')
        return [src]
    report.event(f'split {label} ({src.name}) -> {[o.name for o in outs]}')
    return outs


# ----------------------------------------------------------------------------------------
# stage 1: left / right split (same rule as inference_utils.part_lr_split)
# ----------------------------------------------------------------------------------------

def side_suffix(side: str) -> str:
    return {'character_left': 'L', 'character_right': 'R'}[side]


def split_lr_cc(part: Part, base: Optional[str] = None) -> List[Part]:
    """Existing tblr_split rule on a full canvas.

    Like ``part_lr_split``: take the two largest 8-connected components of alpha>10; the one whose
    bbox centre is further LEFT on screen is the CHARACTER RIGHT (existing name ``<tag>-r``).
    Difference (deliberate, documented): remaining small components and faint (alpha<=10) edge
    pixels are assigned to the nearer of the two instead of being dropped, so nothing is lost.
    """
    comps = components(part.mask)
    if len(comps) < 2:
        return [part]
    (m1, s1, _), (m2, s2, _) = comps[0], comps[1]
    x1 = s1[0] + s1[2] / 2
    x2 = s2[0] + s2[2] / 2
    screen_left, screen_right = (m1, m2) if x1 <= x2 else (m2, m1)
    region = part.alpha > 0
    keep_r, keep_l = assign_to_nearest(region, [screen_left, screen_right])
    b = base or part.name
    return [part.derive(f'{b}_R', keep_r, side='character_right', method='lr_cc'),
            part.derive(f'{b}_L', keep_l, side='character_left', method='lr_cc')]


def split_lr_midline(part: Part, cx: float, base: Optional[str] = None) -> List[Part]:
    """Split a single connected region (e.g. tights) at the body mid-line x=cx."""
    region = part.alpha > 0
    xs = np.arange(part.img.shape[1])[None, :]
    b = base or part.name
    return [part.derive(f'{b}_R', region & (xs < cx), side='character_right', method='lr_midline'),
            part.derive(f'{b}_L', region & (xs >= cx), side='character_left', method='lr_midline')]


def stage_lr_split(parts: List[Part], report: SplitReport, tags=EXISTING_LR_TAGS) -> List[Part]:
    out = []
    for p in parts:
        if p.source in tags and p.side is None:
            out.extend(safe_split(p, split_lr_cc, report, 'lr_split'))
        else:
            out.append(p)
    return out


# ----------------------------------------------------------------------------------------
# canvas mapping: See-through inference canvas (padded square, resolution R) <-> original image
# ----------------------------------------------------------------------------------------

def pad_info(orig_hw, resolution):
    """Reproduce utils.cv.center_square_pad_resize padding for an (H, W) image."""
    h, w = orig_hw
    sz = max(h, w)
    return sz, ((sz - w) // 2, (sz - h) // 2)


def resize_rgba_premultiplied(img: np.ndarray, size_hw) -> np.ndarray:
    """Resize RGBA without colour fringes (premultiplied interpolation)."""
    th, tw = size_hw
    h, w = img.shape[:2]
    if (h, w) == (th, tw):
        return img.copy()
    interp = cv2.INTER_AREA if th * tw < h * w else cv2.INTER_LINEAR
    a = img[..., 3:4].astype(np.float32) / 255.
    pm = np.concatenate([img[..., :3].astype(np.float32) * a, a * 255.], -1)
    r = cv2.resize(pm, (tw, th), interpolation=interp)
    ra = np.clip(r[..., 3:4], 0, 255)
    rgb = np.where(ra > 0.5, r[..., :3] / np.maximum(ra / 255., 1e-6), 0)
    out = np.concatenate([np.clip(rgb, 0, 255), ra], -1)
    return np.round(out).astype(np.uint8)


def to_original_canvas(img: np.ndarray, orig_hw) -> np.ndarray:
    """Map a full-canvas layer from the inference canvas (R x R) back onto the original (H, W) canvas."""
    h, w = orig_hw
    sz, (px, py) = pad_info(orig_hw, img.shape[0])
    sq = resize_rgba_premultiplied(img, (sz, sz))
    return sq[py: py + h, px: px + w].copy()


def from_original_canvas(img: np.ndarray, resolution: int) -> np.ndarray:
    """Original (H, W) -> inference canvas (R x R); inverse of to_original_canvas."""
    h, w = img.shape[:2]
    sz, (px, py) = pad_info((h, w), resolution)
    sq = np.zeros((sz, sz, img.shape[2]), img.dtype)
    sq[py: py + h, px: px + w] = img
    return resize_rgba_premultiplied(sq, (resolution, resolution)) if img.shape[2] == 4 else cv2.resize(sq, (resolution, resolution))


def composite(parts: Sequence[Part], shape=None) -> np.ndarray:
    """Straight-alpha 'over' composite of parts in the given order (first = bottom)."""
    if shape is None:
        shape = parts[0].img.shape
    rgb = np.zeros(shape[:2] + (3,), np.float32)
    a = np.zeros(shape[:2] + (1,), np.float32)
    for p in parts:
        pa = p.img[..., 3:4].astype(np.float32) / 255.
        prgb = p.img[..., :3].astype(np.float32)
        na = pa + a * (1 - pa)
        rgb = np.where(na > 0, (prgb * pa + rgb * a * (1 - pa)) / np.maximum(na, 1e-6), 0)
        a = na
    return np.round(np.concatenate([rgb, a * 255], -1)).clip(0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------
# limb analysis (shape based; no landmark model required)
# ----------------------------------------------------------------------------------------

def geodesic_distance(mask: np.ndarray, seed: np.ndarray, max_steps: int = 100000) -> np.ndarray:
    """Geodesic (inside-mask) distance from ``seed`` by iterative constrained dilation on the ROI.

    Uses alternating 3x3 cross / square kernels (octagonal metric ~ Euclidean).  Pixels not
    reachable get +inf.  Returned array has the full canvas shape (float32).
    """
    x1, y1, x2, y2 = bbox(mask)
    m = mask[y1:y2, x1:x2].astype(np.uint8)
    cur = (seed[y1:y2, x1:x2] & mask[y1:y2, x1:x2]).astype(np.uint8)
    dist = np.full(m.shape, np.inf, np.float32)
    dist[cur > 0] = 0
    k_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    k_sq = np.ones((3, 3), np.uint8)
    step = 0
    while step < max_steps:
        step += 1
        nxt = cv2.dilate(cur, k_sq if step % 2 else k_cross) & m
        new = (nxt > 0) & ~np.isfinite(dist)
        if not new.any():
            break
        dist[new] = step * 1.0
        cur = nxt
    full = np.full(mask.shape, np.inf, np.float32)
    full[y1:y2, x1:x2] = dist
    return full


@dataclass
class LimbAxis:
    """Centre-line of a limb parametrised by geodesic distance from its proximal end."""
    dist: np.ndarray                 # full-canvas geodesic distance (inf outside)
    length: float
    centers: np.ndarray              # (N, 2) xy centre of each distance bin
    widths: np.ndarray               # (N,) pixel count per bin (proxy for width)
    bin_size: float
    proximal: Tuple[float, float]
    distal: Tuple[float, float]

    def point_at(self, d: float) -> Tuple[float, float]:
        i = int(np.clip(d / self.bin_size, 0, len(self.centers) - 1))
        return tuple(self.centers[i])


def limb_axis(mask: np.ndarray, anchor_xy: Tuple[float, float], n_bins: int = 64) -> Optional[LimbAxis]:
    """Proximal end = mask pixels closest to ``anchor_xy`` (e.g. shoulder / hip side)."""
    if mask.sum() < 50:
        return None
    ys, xs = np.nonzero(mask)
    d2 = (xs - anchor_xy[0]) ** 2 + (ys - anchor_xy[1]) ** 2
    near = d2 <= d2.min() + max(9.0, 0.002 * d2.max())
    seed = np.zeros_like(mask)
    seed[ys[near], xs[near]] = True
    dist = geodesic_distance(mask, seed)
    fin = np.isfinite(dist) & mask
    L = float(dist[fin].max())
    if L < 8:
        return None
    bs = L / n_bins
    b = np.clip((dist[fin] / bs).astype(int), 0, n_bins - 1)
    fy, fx = np.nonzero(fin)
    cnt = np.bincount(b, minlength=n_bins).astype(np.float32)
    cx = np.bincount(b, weights=fx, minlength=n_bins) / np.maximum(cnt, 1)
    cy = np.bincount(b, weights=fy, minlength=n_bins) / np.maximum(cnt, 1)
    centers = np.stack([cx, cy], 1)
    # fill empty bins
    for i in range(n_bins):
        if cnt[i] == 0 and i > 0:
            centers[i] = centers[i - 1]
    far = np.argmax(np.where(fin, dist, -1))
    return LimbAxis(dist, L, centers, cnt, bs, tuple(centers[0]), (float(far % mask.shape[1]), float(far // mask.shape[1])))


def _bend_index(axis: LimbAxis, lo: float, hi: float) -> Optional[int]:
    """Bin of maximal deviation from the straight chord between bins lo..hi (fractions)."""
    n = len(axis.centers)
    i0, i1 = int(lo * n), max(int(hi * n), int(lo * n) + 2)
    p0, p1 = axis.centers[i0], axis.centers[min(i1, n - 1)]
    v = p1 - p0
    nv = np.linalg.norm(v)
    if nv < 1e-6:
        return None
    seg = axis.centers[i0:i1]
    dd = seg - p0
    dev = np.abs(v[0] * dd[:, 1] - v[1] * dd[:, 0]) / nv
    j = int(np.argmax(dev))
    return i0 + j if dev[j] > 0.04 * nv else None


def _min_width_index(axis: LimbAxis, lo: float, hi: float) -> int:
    n = len(axis.widths)
    i0, i1 = int(lo * n), max(int(hi * n), int(lo * n) + 1)
    w = np.convolve(axis.widths, np.ones(3) / 3, mode='same')
    return i0 + int(np.argmin(w[i0:i1]))


def arm_joints(axis: LimbAxis, has_hand=True) -> Dict[str, float]:
    """Geodesic positions of elbow / wrist along an arm (shoulder = 0).

    Anthropometric prior: upper arm ~= forearm, hand ~= 0.2 of total; refined by the
    narrowest section near the wrist and by the bend (if the arm is bent) for the elbow.
    """
    L = axis.length
    n = len(axis.centers)
    if has_hand:
        wi = _min_width_index(axis, 0.68, 0.86)
        d_wrist = (wi + 0.5) * axis.bin_size
    else:
        d_wrist = L
    bi = _bend_index(axis, 0.0, min(1.0, d_wrist / L))
    d_elbow = 0.5 * d_wrist
    if bi is not None:
        d_b = (bi + 0.5) * axis.bin_size
        if 0.38 * d_wrist <= d_b <= 0.62 * d_wrist:
            d_elbow = d_b
    return {'elbow': float(d_elbow), 'wrist': float(d_wrist), 'length': float(L), 'n_bins': n}


def leg_joints(axis: LimbAxis, has_foot=False) -> Dict[str, float]:
    """Knee / ankle along a leg (hip = 0). Thigh ~= lower leg; foot ~= 0.12 if included."""
    L = axis.length
    d_ankle = L if not has_foot else (_min_width_index(axis, 0.8, 0.94) + 0.5) * axis.bin_size
    bi = _bend_index(axis, 0.0, min(1.0, d_ankle / L))
    d_knee = 0.5 * d_ankle
    if bi is not None:
        d_b = (bi + 0.5) * axis.bin_size
        if 0.4 * d_ankle <= d_b <= 0.6 * d_ankle:
            d_knee = d_b
    return {'knee': float(d_knee), 'ankle': float(d_ankle), 'length': float(L)}


def segment_with_overlap(part_region: np.ndarray, dist: np.ndarray, cuts: Sequence[float], ov: float) -> List[np.ndarray]:
    """Split a limb at geodesic distances ``cuts`` (ascending). Segment k covers
    [cut_{k-1} - ov, cut_k + ov] so that neighbouring segments OVERLAP by 2*ov around each joint."""
    edges = [-np.inf] + list(cuts) + [np.inf]
    d = np.where(np.isfinite(dist), dist, np.inf)
    out = []
    for k in range(len(edges) - 1):
        lo = edges[k] - ov if np.isfinite(edges[k]) else -np.inf
        hi = edges[k + 1] + ov if np.isfinite(edges[k + 1]) else np.inf
        out.append(part_region & (d >= lo) & (d <= hi))
    # unreachable pixels (inf) -> last segment only if hi is inf, already handled; ensure no loss
    unreached = part_region & ~np.isfinite(dist)
    if unreached.any():
        seeds = [o & np.isfinite(dist) for o in out]
        if all(s.any() for s in seeds):
            for o, extra in zip(out, assign_to_nearest(unreached, seeds)):
                o |= extra
    return out


# ----------------------------------------------------------------------------------------
# stage 2a: additional left/right splits (connected components where appropriate)
# ----------------------------------------------------------------------------------------

EXTRA_CC_LR_TAGS = ['footwear', 'earwear']
SUSPICIOUS_AREA_RATIO = 0.35  # a body-part layer covering >35% of the canvas is likely background leakage


def body_center_x(parts: Sequence[Part]) -> Optional[float]:
    for tag in ('neck', 'face', 'topwear'):
        for p in parts:
            if p.source == tag and p.side is None and p.area() > 0:
                ys, xs = np.nonzero(p.mask)
                return float(np.median(xs))
    return None


def is_suspicious(p: Part) -> bool:
    return p.area() > SUSPICIOUS_AREA_RATIO * p.img.shape[0] * p.img.shape[1]


def stage_lr_extra(parts: List[Part], report: SplitReport) -> List[Part]:
    cx = body_center_x(parts)
    out = []
    for p in parts:
        if p.side is not None:
            out.append(p)
        elif p.source in EXTRA_CC_LR_TAGS:
            out.extend(safe_split(p, split_lr_cc, report, 'lr_cc'))
        elif p.source == 'legwear':
            if is_suspicious(p):
                report.event(f'fallback lr legwear: layer covers {p.area() / p.mask.size:.0%} of canvas (background leakage?) -> kept')
                out.append(p)
                continue

            def _legs(q):
                r = split_lr_cc(q)
                if len(r) == 2:
                    return r
                return split_lr_midline(q, cx) if cx is not None else None
            out.extend(safe_split(p, _legs, report, 'lr_legs'))
        else:
            out.append(p)
    return out


# ----------------------------------------------------------------------------------------
# pipeline
# ----------------------------------------------------------------------------------------

DETAILED_STAGES: List[Tuple[str, Callable]] = []   # (label, fn(parts, report, ctx) -> parts); filled below


@dataclass
class SplitContext:
    canvas_hw: Tuple[int, int]
    overlap_ratio: float = 0.02
    body_cx: Optional[float] = None
    fullpage: Optional[np.ndarray] = None
    arms: Dict[str, dict] = field(default_factory=dict)   # side -> {'axis': LimbAxis, 'joints': {...}}
    legs: Dict[str, dict] = field(default_factory=dict)

    @property
    def ov(self) -> int:
        return overlap_px(self.canvas_hw, self.overlap_ratio)


def run_detailed_split(srcd: str, original: Optional[str] = None, out_psd: Optional[str] = None,
                       overlap_ratio: float = 0.02, restore_visible: bool = True, preview: bool = False,
                       use_groups: bool = True) -> dict:
    """Live2D detailed split of a See-through output directory ``srcd`` (post-processing only).

    original: path of the ORIGINAL input image. If given, parts are exported on its canvas
              (same size / coordinates as the original); otherwise on the inference canvas.
    Returns {'psd', 'counts', 'names', 'preview', 'events'}.
    """
    from .live2d_psd import save_live2d_psd
    report = SplitReport()
    fullpage, parts = load_semantic_layers(srcd)
    report.stage('original semantic layers', len(parts))
    parts = stage_lr_split(parts, report)
    report.stage('after LR split', len(parts))
    ctx = SplitContext(canvas_hw=fullpage.shape[:2], overlap_ratio=overlap_ratio,
                       body_cx=body_center_x(parts), fullpage=fullpage)
    parts = stage_lr_extra(parts, report)
    for label, fn in DETAILED_STAGES:
        try:
            parts = fn(parts, report, ctx)
        except Exception as e:  # noqa: BLE001 - a broken stage must never break the PSD
            report.event(f'stage {label} failed ({type(e).__name__}: {e}); parts left unchanged')
    parts = [p for p in parts if p.area() > 0]
    report.stage('after detailed split', len(parts))

    orig_img = None
    if original is not None:
        orig_img = np.array(Image.open(original).convert('RGBA'))
        hw = orig_img.shape[:2]
        parts = [Part(p.name, to_original_canvas(p.img, hw), p.depth, p.source, p.group, p.side, p.method)
                 for p in parts]
        parts = [p for p in parts if p.area() > 0]
    canvas_hw = parts[0].img.shape[:2] if parts else fullpage.shape[:2]
    # stable sort far -> near (ties keep split order)
    parts = [p for _, _, p in sorted(((-p.depth, i, p) for i, p in enumerate(parts)), key=lambda t: (t[0], t[1]))]
    if restore_visible and orig_img is not None:
        parts = restore_visible_pixels(parts, orig_img, report)

    names = dedupe_names(parts)
    if out_psd is None:
        out_psd = srcd.rstrip('/\\') + '_live2d.psd'
    save_live2d_psd(out_psd, parts, canvas_hw, use_groups=use_groups)
    report.stage('final PSD layers', len(parts))
    prev = None
    if preview and 'make_preview' in globals():
        prev = make_preview(parts, orig_img if orig_img is not None else to_canvas_like(fullpage, canvas_hw),  # noqa: F821
                            osp.splitext(out_psd)[0] + '_preview.png')
    return {'psd': out_psd, 'counts': report.stages, 'names': names, 'preview': prev, 'events': report.events,
            'canvas_hw': list(canvas_hw)}


def to_canvas_like(img, hw):
    return img if img.shape[:2] == tuple(hw) else to_original_canvas(img, hw)


def dedupe_names(parts: List[Part]) -> List[str]:
    seen: Dict[str, int] = {}
    for p in parts:
        if p.name in seen:
            seen[p.name] += 1
            p.name = f'{p.name}_{seen[p.name]:02d}'
        else:
            seen[p.name] = 1
    return [p.name for p in parts]


# ----------------------------------------------------------------------------------------
# stage: arms (handwear L/R) -> upper_arm / forearm / hand (+ sleeve cloth inside handwear)
# ----------------------------------------------------------------------------------------

def shoulder_anchor(parts: Sequence[Part], ctx: 'SplitContext') -> Optional[Tuple[float, float]]:
    """Neck base (bottom-centre of the neck layer, or chin of the face) used as proximal anchor."""
    for tag in ('neck', 'face'):
        for p in parts:
            if p.source == tag and p.area() > 0:
                ys, xs = np.nonzero(p.mask)
                return float(np.median(xs)), float(ys.max())
    return None


def _lab(img_rgb_pixels: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def cloth_skin_masks(part: Part, region: np.ndarray, hand_core: np.ndarray, min_chroma=10.0, min_frac=0.06):
    """Separate cloth (sleeve) from skin inside an arm layer by CHROMA (Lab a,b) difference to the hand.

    Lightness is ignored on purpose so that shading / shadows on a bare arm are not mistaken for cloth.
    A pixel is cloth if its (a,b) distance to the median hand colour exceeds
    max(min_chroma, 3 * the hand's own chroma spread).  Small blobs are removed.  Returns None if no
    coherent cloth region exists (then the arm is not split into skin / sleeve)."""
    if hand_core.sum() < 20 or region.sum() < 200:
        return None
    ab = cv2.cvtColor(part.img[..., :3], cv2.COLOR_RGB2LAB)[..., 1:].astype(np.float32)
    hand_ab = ab[hand_core]
    ref = np.median(hand_ab, 0)
    spread = np.percentile(np.linalg.norm(hand_ab - ref, axis=1), 90)
    thr = max(min_chroma, 3.0 * spread)
    dist = np.linalg.norm(ab - ref, axis=-1)
    cloth = (dist > thr) & region & part.mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cloth = cv2.morphologyEx(cloth.astype(np.uint8), cv2.MORPH_OPEN, k)
    cloth = cv2.morphologyEx(cloth, cv2.MORPH_CLOSE, k).astype(bool) & region
    comps = components(cloth, min_area=max(30, int(0.03 * region.sum())))
    cloth = np.zeros_like(cloth)
    for c, _, _ in comps:
        cloth |= c
    if cloth.sum() < min_frac * region.sum():
        return None
    # include the faint (alpha<=10) rim around cloth so nothing is orphaned
    rim = cv2.dilate(cloth.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & region & ~part.mask
    return cloth | rim


def split_arm(part: Part, anchor, ctx: 'SplitContext') -> List[Part]:
    sfx = side_suffix(part.side)
    m = part.mask
    axis = limb_axis(m, anchor)
    if axis is None:
        return None
    if axis.length < 0.08 * long_side(m.shape):  # only a hand / glove
        return [part.derive(f'hand_{sfx}', part.alpha > 0, side=part.side, method='arm_hand_only')]
    j = arm_joints(axis)
    ov = ctx.ov
    region = part.alpha > 0
    upper, fore, hand = segment_with_overlap(region, axis.dist, [j['elbow'], j['wrist']], ov)
    hand_core = m & np.isfinite(axis.dist) & (axis.dist > j['wrist'] + ov)
    cloth = cloth_skin_masks(part, region & ~hand_core, hand_core)
    outs = []
    pieces = [('upper_arm', 'upper_sleeve', upper), ('forearm', 'lower_sleeve', fore)]
    min_px = max(16, int(0.01 * m.sum()))
    for skin_name, cloth_name, seg in pieces:
        if cloth is None:
            outs.append(part.derive(f'{skin_name}_{sfx}', seg, side=part.side, method='arm_geodesic'))
            continue
        c = seg & cloth
        s = seg & ~cloth
        if (c & m).sum() < min_px:
            s, c = seg, None
        elif (s & m).sum() < min_px:
            s, c = None, seg
        if s is not None:
            outs.append(part.derive(f'{skin_name}_{sfx}', s, side=part.side, method='arm_geodesic'))
        if c is not None:
            sl = part.derive(f'{cloth_name}_{sfx}', c, side=part.side, method='arm_sleeve_colour', group='Clothes')
            sl.depth = part.depth - 1e-3
            outs.append(sl)
    outs.append(part.derive(f'hand_{sfx}', hand, side=part.side, method='arm_geodesic'))
    for o in outs:
        o.depth_map = None
    ctx.arms[part.side] = {'axis': axis, 'joints': j, 'mask': m}
    return outs


def stage_arms(parts: List[Part], report: SplitReport, ctx: 'SplitContext') -> List[Part]:
    anchor = shoulder_anchor(parts, ctx)
    out = []
    for p in parts:
        if p.source == 'handwear' and p.side is not None and anchor is not None:
            out.extend(safe_split(p, lambda q: split_arm(q, anchor, ctx), report, 'arm', min_cover=0.995))
        else:
            out.append(p)
    return out


DETAILED_STAGES.append(('arms', stage_arms))


# ----------------------------------------------------------------------------------------
# stage: topwear -> torso / upper_sleeve / lower_sleeve / cuff (uses the arm axes)
# ----------------------------------------------------------------------------------------

def _nearest_on_polyline(ys, xs, pts):
    """For pixels (ys, xs): index of nearest polyline vertex and distance (vectorised, chunked)."""
    idx = np.empty(len(xs), np.int64)
    dist = np.empty(len(xs), np.float32)
    for s0 in range(0, len(xs), 20000):
        dx = xs[s0:s0 + 20000, None] - pts[None, :, 0]
        dy = ys[s0:s0 + 20000, None] - pts[None, :, 1]
        d2 = dx * dx + dy * dy
        k = np.argmin(d2, 1)
        idx[s0:s0 + 20000] = k
        dist[s0:s0 + 20000] = np.sqrt(d2[np.arange(len(k)), k])
    return idx, dist


def _median_ab(img, m):
    return np.median(cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2LAB)[..., 1:][m].astype(np.float32), 0)


def split_topwear(part: Part, ctx: 'SplitContext', anchor) -> List[Part]:
    if not ctx.arms or ctx.body_cx is None or anchor is None:
        return None
    region = part.alpha > 0
    ys, xs = np.nonzero(region)
    ov = ctx.ov
    # torso skeleton: vertical segment at body centre from the neck base to the bottom of the layer
    y_top, y_bot = anchor[1], ys.max()
    rows = np.unique(ys)
    mid_rows = rows[(rows > y_top + 0.2 * (y_bot - y_top)) & (rows < y_top + 0.8 * (y_bot - y_top))]
    xc = int(round(ctx.body_cx))
    half = []
    for r in mid_rows[::4]:
        row = region[r]
        if row[xc]:
            l = xc
            while l > 0 and row[l - 1]:
                l -= 1
            rr = xc
            while rr < len(row) - 1 and row[rr + 1]:
                rr += 1
            half.append((rr - l) / 2)
    if not half:
        return None
    torso_r = float(np.median(half))
    # waist half-width (lower rows): sleeves must protrude beyond the torso silhouette
    low_rows = rows[rows > y_top + 0.6 * (y_bot - y_top)]
    waist = []
    for r in low_rows[::3]:
        row = region[r]
        if row[xc]:   # contiguous run through the body centre (excludes separate hanging sleeves)
            l = xc
            while l > 0 and row[l - 1]:
                l -= 1
            rr = xc
            while rr < len(row) - 1 and row[rr + 1]:
                rr += 1
            waist.append(max(xc - l, rr - xc))
    waist_r = float(np.median(waist)) if waist else torso_r
    outside_torso = np.abs(xs - ctx.body_cx) > 1.05 * waist_r
    torso_pts = np.stack([np.full(64, ctx.body_cx), np.linspace(y_top, y_bot, 64)], 1)
    _, d_torso = _nearest_on_polyline(ys, xs, torso_pts)
    scores = [d_torso / max(torso_r, 1.0)]
    sides = []
    for side, a in ctx.arms.items():
        ax, j = a['axis'], a['joints']
        bins = np.arange(len(ax.centers))
        dbin = (bins + 0.5) * ax.bin_size
        keep = dbin >= 0.15 * j['elbow']
        pts = ax.centers[keep]
        if len(pts) < 3:
            continue
        w = ax.widths[keep & (dbin <= j['elbow'])] / ax.bin_size
        arm_r = float(np.median(w)) / 2 if len(w) else 10.0
        k, d_arm = _nearest_on_polyline(ys, xs, pts)
        sc = d_arm / max(arm_r * 1.6, 1.0)   # sleeves are wider than the bare arm
        # a sleeve must lie OVER the (inpainted) arm: pixels far outside the arm silhouette stay torso
        near_arm = cv2.dilate(a['mask'].astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * max(2, int(0.15 * arm_r)) + 1,) * 2)).astype(bool)
        sc = np.where(near_arm[ys, xs] & outside_torso, sc, np.inf)
        scores.append(sc)
        sides.append((side, dbin[keep][k]))
    if not sides:
        return None
    owner = np.argmin(np.stack(scores, 0), 0)        # 0 = torso, i = sides[i-1]
    lab = np.full(region.shape, -1, np.int32)
    lab[ys, xs] = owner
    outs = []
    torso = lab == 0
    outs.append(part.derive('torso', torso, method='topwear_skeleton'))
    for i, (side, dpix) in enumerate(sides, start=1):
        sleeve = lab == i
        sm = sleeve & part.mask
        arm_m = ctx.arms[side]['mask']
        over_arm = (sm & arm_m).sum() / max(sm.sum(), 1)
        along = dpix[owner == i]
        reach = (np.percentile(along, 95) - np.percentile(along, 5)) / max(ctx.arms[side]['joints']['elbow'], 1) if len(along) else 0
        if sm.sum() < max(50, 0.02 * part.area()) or over_arm < 0.6 or reach < 0.4:
            torso |= sleeve   # not a real sleeve (e.g. off-shoulder / sleeveless top next to the arm)
            continue
        sfx = side_suffix(side)
        j = ctx.arms[side]['joints']
        dmap = np.full(region.shape, np.inf, np.float32)
        dmap[ys, xs] = dpix
        # sleeve keeps an overlap band into the torso at the shoulder seam
        ext = cv2.dilate(sleeve.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ov + 1,) * 2)).astype(bool)
        sleeve_ext = ext & region & (sleeve | torso)
        dmap[sleeve_ext & ~sleeve] = 0
        upper, lower = segment_with_overlap(sleeve_ext, dmap, [j['elbow']], ov)
        pieces = [('upper_sleeve', upper), ('lower_sleeve', lower)]
        # cuff: distal band with clearly different colour from the sleeve body
        dmax = float(dpix[owner == i].max())
        band = sleeve & (dmap >= dmax - 0.15 * (dmax - j['elbow'])) & part.mask
        body = sleeve & (dmap < dmax - 0.3 * (dmax - j['elbow'])) & part.mask
        if band.sum() > 30 and body.sum() > 30 and np.linalg.norm(_median_ab(part.img, band) - _median_ab(part.img, body)) > 10:
            cut = dmax - 0.15 * (dmax - j['elbow'])
            lower2, cuff = segment_with_overlap(lower, dmap, [cut], max(2, ov // 2))
            pieces = [('upper_sleeve', upper), ('lower_sleeve', lower2), ('cuff', cuff)]
        if len(pieces) == 2 and (lower & ~upper & part.mask).sum() < 0.15 * sm.sum():
            pieces = [('upper_sleeve', upper | lower)]   # sleeve ends above/near the elbow: one piece
        for nm, m in pieces:
            if (m & part.mask).sum() >= 16:
                outs.append(part.derive(f'{nm}_{sfx}', m, side=side, method='topwear_skeleton'))
    if len(outs) == 1:
        return None   # no real sleeve found -> keep topwear as is
    outs[0] = part.derive('torso', torso, method='topwear_skeleton')
    return outs


def stage_topwear(parts: List[Part], report: SplitReport, ctx: 'SplitContext') -> List[Part]:
    anchor = shoulder_anchor(parts, ctx)
    out = []
    for p in parts:
        if p.source == 'topwear' and p.side is None:
            out.extend(safe_split(p, lambda q: split_topwear(q, ctx, anchor), report, 'topwear'))
        else:
            out.append(p)
    return out


DETAILED_STAGES.append(('topwear', stage_topwear))


# ----------------------------------------------------------------------------------------
# stage: legs (legwear L/R) -> thigh / lower_leg (/ foot when there is no footwear layer)
# ----------------------------------------------------------------------------------------

def split_leg(part: Part, ctx: 'SplitContext', has_footwear: bool) -> List[Part]:
    sfx = side_suffix(part.side)
    m = part.mask
    ys, xs = np.nonzero(m)
    y0 = ys.min()
    top = ys <= y0 + max(2, 0.03 * (ys.max() - y0))
    anchor = (float(np.median(xs[top])), float(y0) - 1)
    axis = limb_axis(m, anchor)
    if axis is None or axis.length < 0.12 * long_side(m.shape):
        return None
    j = leg_joints(axis, has_foot=not has_footwear)
    ov = ctx.ov
    region = part.alpha > 0
    cuts = [j['knee']] + ([j['ankle']] if not has_footwear else [])
    segs = segment_with_overlap(region, axis.dist, cuts, ov)
    names = ['thigh', 'lower_leg', 'foot'][:len(segs)]
    ctx.legs[part.side] = {'axis': axis, 'joints': j, 'mask': m}
    return [part.derive(f'{n}_{sfx}', s, side=part.side, method='leg_geodesic') for n, s in zip(names, segs)]


def stage_legs(parts: List[Part], report: SplitReport, ctx: 'SplitContext') -> List[Part]:
    has_fw = any(p.source == 'footwear' for p in parts)
    out = []
    for p in parts:
        if p.source == 'legwear' and p.side is not None:
            out.extend(safe_split(p, lambda q: split_leg(q, ctx, has_fw), report, 'leg'))
        else:
            out.append(p)
    return out


DETAILED_STAGES.append(('legs', stage_legs))


# ----------------------------------------------------------------------------------------
# stage: hair strands (shape analysis; NOT connected-components only)
# ----------------------------------------------------------------------------------------

@dataclass
class FaceInfo:
    cx: float
    cy: float
    top: float
    bottom: float
    half_w: float
    eye_y: float


def face_info(parts: Sequence[Part]) -> Optional[FaceInfo]:
    face = next((p for p in parts if p.source == 'face' and p.area() > 0), None)
    if face is None:
        return None
    ys, xs = np.nonzero(face.mask)
    eyes = [p for p in parts if p.source in ('eyewhite', 'irides', 'eyes') and p.area() > 0]
    eye_y = float(np.mean([np.nonzero(p.mask)[0].mean() for p in eyes])) if eyes else float(ys.min() + 0.45 * np.ptp(ys))
    return FaceInfo(float(np.median(xs)), float(ys.mean()), float(ys.min()), float(ys.max()),
                    float(np.percentile(xs, 99.5) - np.percentile(xs, 0.5)) / 2, eye_y)


def _angle_valley_cuts(theta, radius, targets, lo, hi, window, nbins=180):
    """Snap each target cut angle to the angle in [t-window, t+window] where the hair is SHORTEST
    (max radius from the pivot is minimal) - i.e. the gap between two strands."""
    if len(theta) == 0:
        return list(targets)
    edges = np.linspace(lo, hi, nbins + 1)
    b = np.clip(np.digitize(theta, edges) - 1, 0, nbins - 1)
    rmax = np.zeros(nbins, np.float32)
    np.maximum.at(rmax, b, radius.astype(np.float32))
    rmax = np.convolve(rmax, np.ones(3) / 3, mode='same')
    centers = (edges[:-1] + edges[1:]) / 2
    out = []
    for t in targets:
        sel = (centers >= t - window) & (centers <= t + window) & (rmax > 0)
        if not sel.any():
            out.append(t)
            continue
        cand = np.nonzero(sel)[0]
        k = cand[np.argmin(rmax[cand] + 1e-3 * np.abs(centers[cand] - t))]
        out.append(float(centers[k]))
    return sorted(out)


def _sectors(region, theta_map, cuts, soft_px):
    """Bool masks for angle sectors split at ``cuts``; each sector is grown by ``soft_px`` into its
    neighbours (small overlap so strands do not open gaps when moved)."""
    edges = [-np.inf] + list(cuts) + [np.inf]
    out = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * soft_px + 1,) * 2) if soft_px > 0 else None
    for a, b in zip(edges[:-1], edges[1:]):
        m = region & (theta_map >= a) & (theta_map < b)
        if k is not None and m.any():
            m = cv2.dilate(m.astype(np.uint8), k).astype(bool) & region
        out.append(m)
    return out


def split_front_hair(part: Part, fi: FaceInfo, ctx: 'SplitContext') -> List[Part]:
    region = part.alpha > 0
    H, W = region.shape
    ys, xs = np.nonzero(part.mask)
    yy, xx = np.mgrid[0:H, 0:W]
    outs = []
    # side locks: hair outside the face width hanging below the eye line
    side_band = (np.abs(xx - fi.cx) > 0.92 * fi.half_w) & (yy > fi.eye_y - 0.1 * (fi.bottom - fi.top))
    side = region & side_band
    rest = region.copy()
    for nm, sel in (('side_hair_R', xx < fi.cx), ('side_hair_L', xx >= fi.cx)):
        m = side & sel
        # keep only side hair pieces that are connected to strands hanging down (not stray pixels)
        if (m & part.mask).sum() >= 0.03 * part.area():
            rest &= ~m
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, ctx.ov // 2) + 1,) * 2)
            m = cv2.dilate(m.astype(np.uint8), k).astype(bool) & region   # overlap into the bangs (no gap)
            outs.append(part.derive(nm, m, side='character_right' if nm.endswith('R') else 'character_left',
                                    method='hair_side'))
    # bangs: radial sectors around a pivot above the forehead, cut in the gaps between strands
    py = fi.top - 0.35 * (fi.bottom - fi.top)
    px = fi.cx
    theta_map = np.arctan2(xx - px, yy - py)        # 0 = straight down, <0 = screen left
    r_map = np.hypot(xx - px, yy - py)
    bm = rest & part.mask
    th, rr = theta_map[bm], r_map[bm]
    if len(th) < 50:
        return outs + [part.derive('front_hair_center', rest, method='hair_sector')] if outs else None
    width_ratio = (np.percentile(xx[bm], 97) - np.percentile(xx[bm], 3)) / max(2 * fi.half_w, 1)
    if width_ratio > 0.6:
        qs = [0.2, 0.38, 0.62, 0.8]
        names = ['front_hair_R_02', 'front_hair_R_01', 'front_hair_center', 'front_hair_L_01', 'front_hair_L_02']
    else:
        qs = [0.33, 0.67]
        names = ['front_hair_R_01', 'front_hair_center', 'front_hair_L_01']
    targets = list(np.quantile(th, qs))
    win = 0.4 * float(np.min(np.diff([th.min()] + targets + [th.max()])))
    cuts = _angle_valley_cuts(th, rr, targets, th.min(), th.max(), win)
    soft = max(1, ctx.ov // 4)
    for nm, m in zip(names, _sectors(rest, theta_map, cuts, soft)):
        sd = 'character_right' if '_R_' in nm else 'character_left' if '_L_' in nm else None
        outs.append(part.derive(nm, m, side=sd, method='hair_sector'))
    return outs


def split_back_hair(part: Part, fi: FaceInfo, ctx: 'SplitContext') -> List[Part]:
    region = part.alpha > 0
    H, W = region.shape
    yy, xx = np.mgrid[0:H, 0:W]
    px, py = fi.cx, fi.cy
    center = region & (np.abs(xx - px) < 0.35 * fi.half_w)
    outs = [part.derive('back_hair_center', center, method='hair_sector')]
    theta_map = np.abs(np.arctan2(xx - px, yy - py))   # 0 = down, pi = up (per side)
    r_map = np.hypot(xx - px, yy - py)
    soft = max(1, ctx.ov // 4)
    for sfx, sel, sd in (('R', xx < px, 'character_right'), ('L', xx >= px, 'character_left')):
        side = region & sel & ~center
        sm = side & part.mask
        if sm.sum() < 0.04 * part.area():
            if side.any():
                outs.append(part.derive(f'back_hair_{sfx}_01', side, side=sd, method='hair_sector'))
            continue
        th, rr = theta_map[sm], r_map[sm]
        n = 3 if sm.sum() > 0.12 * part.area() else 2
        targets = list(np.quantile(th, np.linspace(0, 1, n + 1)[1:-1]))
        win = 0.35 * float(np.min(np.diff([th.min()] + targets + [th.max()])))
        cuts = _angle_valley_cuts(th, rr, targets, th.min(), th.max(), win)
        for i, m in enumerate(_sectors(side, theta_map, cuts, soft), start=1):   # 01 = lowest (hanging down)
            outs.append(part.derive(f'back_hair_{sfx}_{i:02d}', m, side=sd,
                                    method='hair_sector'))
    return outs


def stage_hair(parts: List[Part], report: SplitReport, ctx: 'SplitContext') -> List[Part]:
    fi = face_info(parts)
    out = []
    for p in parts:
        if fi is not None and p.source == 'front hair':
            out.extend(safe_split(p, lambda q: split_front_hair(q, fi, ctx), report, 'front_hair'))
        elif fi is not None and p.source == 'back hair':
            out.extend(safe_split(p, lambda q: split_back_hair(q, fi, ctx), report, 'back_hair'))
        else:
            out.append(p)
    return out


DETAILED_STAGES.append(('hair', stage_hair))


# ----------------------------------------------------------------------------------------
# stage: bottomwear (skirt -> skirt_R / skirt_center / skirt_L, pants/shorts -> bottomwear_R / _L)
# ----------------------------------------------------------------------------------------

def leg_gap_x(mask: np.ndarray) -> Optional[float]:
    """x of the gap between the legs of shorts / pants (internal hole in the lowest rows), else None."""
    ys, xs = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max()
    width = max(np.ptp(xs), 1)
    gaps = []
    for y in range(int(y1 - 0.1 * (y1 - y0)), y1 + 1):
        cols = np.nonzero(mask[y])[0]
        if len(cols) < 2:
            continue
        d = np.diff(cols)
        k = np.nonzero(d > 1)[0]
        runs = np.split(cols, k + 1)
        big = [r for r in runs if len(r) > 0.12 * width]
        if len(big) >= 2:
            gaps.append((big[0][-1] + big[1][0]) / 2)
    if len(gaps) >= 2:
        return float(np.median(gaps))
    return None


def split_bottomwear(part: Part, ctx: 'SplitContext') -> List[Part]:
    if ctx.body_cx is None:
        return None
    region = part.alpha > 0
    H, W = region.shape
    yy, xx = np.mgrid[0:H, 0:W]
    m = part.mask
    gx = leg_gap_x(m)
    if gx is not None:   # shorts / pants: split between the legs
        r = split_lr_midline(part, gx, base='bottomwear')
        for q in r:
            q.method = 'bottom_pants_gap'
        return r
    ys, xs = np.nonzero(m)
    px, py = ctx.body_cx, ys.min() - 0.3 * np.ptp(ys)      # pivot above the waist (skirt swings from the waist)
    theta = np.arctan2(xx - px, yy - py)
    th, rr = theta[m], np.hypot(xx - px, yy - py)[m]
    targets = list(np.quantile(th, [0.3, 0.7]))
    win = 0.3 * float(np.min(np.diff([th.min()] + targets + [th.max()])))
    cuts = _angle_valley_cuts(th, rr, targets, th.min(), th.max(), win)
    # front/back cannot be determined from a single front view -> left / centre / right
    names = [('skirt_R', 'character_right'), ('skirt_center', None), ('skirt_L', 'character_left')]
    secs = _sectors(region, theta, cuts, max(1, ctx.ov // 4))
    return [part.derive(n, s, side=sd, method='skirt_sector') for (n, sd), s in zip(names, secs)]


def stage_bottomwear(parts, report, ctx):
    out = []
    for p in parts:
        if p.source == 'bottomwear' and p.side is None:
            out.extend(safe_split(p, lambda q: split_bottomwear(q, ctx), report, 'bottomwear'))
        else:
            out.append(p)
    return out


DETAILED_STAGES.append(('bottomwear', stage_bottomwear))


# ----------------------------------------------------------------------------------------
# stage: accessories (headwear / neckwear / objects / earwear / tail / wings)
# ----------------------------------------------------------------------------------------

ACCESSORY_TAGS = ['headwear', 'neckwear', 'objects', 'tail', 'wings', 'eyewear']


def split_ribbon(part: Part, ctx: 'SplitContext') -> Optional[List[Part]]:
    """One connected bow / tie: knot+loops = main, hanging pieces below the knot = tail_R / tail_L."""
    m = part.mask
    ys, xs = np.nonzero(m)
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    xmed = float(np.median(xs))
    upper = m & (np.arange(m.shape[0])[:, None] < ys.min() + 0.6 * np.ptp(ys)) \
        & (np.abs(np.arange(m.shape[1])[None, :] - xmed) < 0.15 * max(np.ptp(xs), 1))
    if not upper.any():
        return None
    k = np.argmax(np.where(upper, dt, -1))
    ky, kx = divmod(int(k), m.shape[1])
    rk = max(float(dt[ky, kx]), 2.0)
    region = part.alpha > 0
    H, W = m.shape
    yy, xx = np.mgrid[0:H, 0:W]
    below = region & (yy > ky + 1.5 * rk)
    tails = []
    for nm, sel, sd in (('tail_R', xx < kx, 'character_right'), ('tail_L', xx >= kx, 'character_left')):
        t = below & sel
        if (t & m).sum() >= 0.1 * m.sum():
            tails.append((nm, t, sd))
    if not tails:
        return None
    main = region.copy()
    for _, t, _ in tails:
        main &= ~t
    b = part.name
    ov = max(1, ctx.ov // 3)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ov + 1,) * 2)
    outs = [part.derive(f'{b}_main', main, method='ribbon_shape')]
    for nm, t, sd in tails:
        t = cv2.dilate(t.astype(np.uint8), kern).astype(bool) & region     # tails overlap the knot
        outs.append(part.derive(f'{b}_{nm}', t, side=sd, method='ribbon_shape'))
    return outs


def split_accessory(part: Part, ctx: 'SplitContext') -> Optional[List[Part]]:
    m = part.mask
    min_a = max(16, int(0.02 * m.sum()))
    comps = [c for c in components(m) if c[1][cv2.CC_STAT_AREA] >= min_a]
    if len(comps) >= 2 and ctx.body_cx is not None:
        region = part.alpha > 0
        seeds = [c[0] for c in comps]
        pieces = assign_to_nearest(region, seeds)
        named = []
        for (cm, st, cen), pc in zip(comps, pieces):
            dx = cen[0] - ctx.body_cx
            tol = 0.04 * m.shape[1]
            if abs(dx) <= tol:
                nm, sd = 'center', None
            elif dx < 0:
                nm, sd = 'R', 'character_right'
            else:
                nm, sd = 'L', 'character_left'
            named.append((nm, sd, pc, cen))
        named.sort(key=lambda t: (t[0], t[3][1]))
        outs, cnt = [], {}
        for nm, sd, pc, _ in named:
            cnt[nm] = cnt.get(nm, 0) + 1
            outs.append((nm, sd, pc, cnt[nm]))
        res = []
        for nm, sd, pc, i in outs:
            suffix = nm if cnt[nm] == 1 else f'{nm}_{i:02d}'
            res.append(part.derive(f'{part.name}_{suffix}', pc, side=sd, method='accessory_cc'))
        return res
    if part.source in ('neckwear', 'headwear'):
        return split_ribbon(part, ctx)
    return None


def stage_accessories(parts, report, ctx):
    out = []
    for p in parts:
        if p.source in ACCESSORY_TAGS and p.side is None:
            out.extend(safe_split(p, lambda q: split_accessory(q, ctx), report, 'accessory'))
        else:
            out.append(p)
    return out


DETAILED_STAGES.append(('accessories', stage_accessories))


# ----------------------------------------------------------------------------------------
# visible-pixel restore: result[visible] = original[visible] (with feather); hidden pixels untouched
# ----------------------------------------------------------------------------------------

def restore_visible_pixels(parts: List[Part], original: np.ndarray, report: Optional[SplitReport] = None,
                           feather_px: Optional[int] = None) -> List[Part]:
    """Copy the ORIGINAL image's RGB into the parts where they are visible in the final composite.

    ``parts`` must be on the original canvas and ordered far -> near.  Visibility is computed per
    semantic source (parts split from the same layer never occlude each other, so joint overlaps
    stay consistent).  Inside a part, weight = 1 deep inside the visible region and ramps to 0 over
    ``feather_px`` pixels toward occlusion boundaries / the part's own soft edge.  Alpha is never
    changed; inpainted (hidden) pixels are kept exactly.
    """
    if not parts:
        return parts
    H, W = parts[0].img.shape[:2]
    if original.shape[:2] != (H, W):
        raise ValueError('original must be on the same canvas as the parts')
    if feather_px is None:
        feather_px = max(2, int(round(0.004 * max(H, W))))
    orig_rgb = original[..., :3].astype(np.float32)
    orig_a = original[..., 3] if original.shape[2] == 4 else np.full((H, W), 255, np.uint8)
    # source blocks in drawing order
    order = []
    for p in parts:
        if not order or order[-1] != p.source:
            order.append(p.source)
    block_alpha = {}
    for s in set(order):
        a = np.zeros((H, W), np.float32)
        for p in parts:
            if p.source == s:
                a = np.maximum(a, p.img[..., 3].astype(np.float32) / 255.)
        block_alpha[s] = a
    # transmittance of everything in front of each block (blocks may repeat if interleaved: use last)
    T = np.ones((H, W), np.float32)
    trans_front = {}
    for s in reversed(order):
        if s not in trans_front:
            trans_front[s] = T.copy()
        T = T * (1 - block_alpha[s])
    n_changed = 0
    for p in parts:
        if p.source in ('nose', 'mouth'):   # further_extr already takes these from the source image
            continue
        a = p.img[..., 3]
        core = (a > 200) & (trans_front[p.source] > 0.95) & (orig_a > 200)
        if not core.any():
            continue
        dist = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 3)
        w = np.clip(dist / float(feather_px), 0, 1)[..., None]
        sel = w[..., 0] > 0
        rgb = p.img[..., :3].astype(np.float32)
        rgb[sel] = w[sel] * orig_rgb[sel] + (1 - w[sel]) * rgb[sel]
        p.img[..., :3] = np.round(rgb).astype(np.uint8)
        n_changed += int(sel.sum())
    if report is not None:
        report.event(f'visible restore: {n_changed} px taken from the original (feather {feather_px}px)')
    return parts
