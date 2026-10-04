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
    dev = np.abs(np.cross(v, seg - p0)) / nv
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
    if restore_visible and orig_img is not None and 'restore_visible_pixels' in globals():
        parts = restore_visible_pixels(parts, orig_img, report)  # noqa: F821 (defined in TASK-017)

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
