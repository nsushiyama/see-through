"""
Fine-grained layer splitting: cut See-through layers into more pieces after decomposition.

LayerDiff 3D only knows the 23 trained body-part tags, so finer layers are produced
heuristically here: each target layer is labelled (position/depth clustering, line-art
watershed, colour clustering or connected components), the labels are ordered front to
back, and the region each front piece hides is inpainted into the pieces behind it so
every piece stays a complete layer.

Only numpy / OpenCV are required (no torch). LaMa can optionally be used for inpainting.

Part dict convention (same as further_extr / psd2partdicts):
    img:   HxWx4 uint8, crop of the canvas at xyxy
    depth: HxW float32 in [0, 1], smaller is closer to the viewer
    xyxy:  [x1, y1, x2, y2] position on the canvas
"""

import json
import os.path as osp

import cv2
import numpy as np


SPLIT_MODES = ['depth_position', 'lineart_watershed', 'color', 'components', 'depth']

ALPHA_THR = 10

_LABEL_KEYS = ('depth_weight', 'color_weight', 'min_area_ratio', 'split_components', 'max_pieces')

PRESET_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'assets', 'fine_split')


def load_fine_split_config(config):
    """config: dict, path to a json file, or a preset name under assets/fine_split."""
    if isinstance(config, dict):
        return config
    if not osp.exists(config):
        presetp = osp.join(PRESET_DIR, config if config.endswith('.json') else config + '.json')
        if not osp.exists(presetp):
            raise FileNotFoundError(f'fine split config not found: {config}')
        config = presetp
    with open(config, 'r', encoding='utf8') as f:
        return json.load(f)


def _norm(x):
    x = x.astype(np.float32)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-6)


def _kmeans(features, k, seed=0, max_samples=6000):
    n = len(features)
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros(n, dtype=np.int32)
    rng = np.random.default_rng(seed)
    samples = features if n <= max_samples else features[rng.choice(n, max_samples, replace=False)]
    cv2.setRNGSeed(int(seed))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    _, _, centers = cv2.kmeans(samples.astype(np.float32), k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    dist = ((features[:, None, :] - centers[None]) ** 2).sum(-1)
    return np.argmin(dist, axis=1).astype(np.int32)


def _cluster_features(img, depth, mask, mode, depth_weight, color_weight):
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    scale = float(max(h, w))
    feats = [xs[:, None] / scale, ys[:, None] / scale]
    if mode in {'depth_position', 'lineart_watershed', 'depth'}:
        d = _norm(depth[mask])[:, None] * depth_weight
        if mode == 'depth':
            return d
        feats.append(d)
    if mode == 'color':
        lab = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2LAB)[mask].astype(np.float32) / 255.
        feats = [f * 0.5 for f in feats] + [lab * color_weight]
    return np.concatenate(feats, axis=1)


def _watershed(img, mask, seed_labels, erode=2):
    """Snap cluster boundaries to line art: seeds are eroded clusters, the rest is flooded on the RGB image."""
    markers = np.zeros(mask.shape, dtype=np.int32)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
    for lid in np.unique(seed_labels[mask]):
        m = cv2.erode((seed_labels == lid).astype(np.uint8), element)
        markers[m > 0] = lid + 1
    a = img[..., 3:4].astype(np.float32) / 255.
    rgb = np.round(img[..., :3].astype(np.float32) * a + 255 * (1 - a)).astype(np.uint8)
    markers[~mask] = -1
    markers = cv2.watershed(np.ascontiguousarray(rgb), markers)
    labels = markers - 1
    labels[~mask] = -1
    # watershed boundary pixels (-1 -> -2) go to the nearest label
    return _fill_unlabeled(labels, mask)


def _fill_unlabeled(labels, mask):
    hole = mask & (labels < 0)
    if not np.any(hole):
        labels[~mask] = -1
        return labels
    known = (mask & (labels >= 0))
    if not np.any(known):
        labels[mask] = 0
        return labels
    _, idx = cv2.distanceTransformWithLabels((~known).astype(np.uint8), cv2.DIST_L2, 5,
                                            labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(known)
    lut = np.full(idx.max() + 1, -1, dtype=np.int32)
    lut[idx[ys, xs]] = labels[ys, xs]
    labels = labels.copy()
    labels[hole] = lut[idx[hole]]
    labels[~mask] = -1
    return labels


def _relabel_compact(labels, mask):
    out = np.full_like(labels, -1)
    for new, lid in enumerate(np.unique(labels[mask])):
        out[labels == lid] = new
    return out


def _split_components(labels, mask):
    out = np.full_like(labels, -1)
    nxt = 0
    for lid in np.unique(labels[mask]):
        n, cc = cv2.connectedComponents((labels == lid).astype(np.uint8), connectivity=8)
        for c in range(1, n):
            out[cc == c] = nxt
            nxt += 1
    return out


def _merge_small(labels, mask, min_area, max_pieces=0):
    """Merge pieces smaller than min_area (and the smallest ones beyond max_pieces) into their largest neighbour."""
    element = np.ones((3, 3), np.uint8)
    while True:
        ids, areas = np.unique(labels[mask], return_counts=True)
        if len(ids) <= 1:
            break
        order = np.argsort(areas)
        victim = None
        if areas[order[0]] < min_area:
            victim = ids[order[0]]
        elif max_pieces > 0 and len(ids) > max_pieces:
            victim = ids[order[0]]
        if victim is None:
            break
        vmask = labels == victim
        ring = cv2.dilate(vmask.astype(np.uint8), element) > 0
        ring &= mask & ~vmask
        neigh = labels[ring]
        if len(neigh) == 0:
            # isolated island: attach to the largest piece so it is not lost
            others = ids[ids != victim]
            target = others[np.argmax(areas[ids != victim])]
        else:
            nids, ncnt = np.unique(neigh, return_counts=True)
            target = nids[np.argmax(ncnt)]
        labels[vmask] = target
    return _relabel_compact(labels, mask)


def compute_labels(img, depth, mode='lineart_watershed', k=4, depth_weight=1.0, color_weight=2.0,
                   min_area_ratio=0.01, split_components=True, max_pieces=0, seed=0):
    """Returns (labels, order): labels HxW int32 (-1 outside), order = label ids front to back."""
    assert mode in SPLIT_MODES, f'invalid split mode {mode}, valid: {SPLIT_MODES}'
    mask = img[..., 3] > ALPHA_THR
    labels = np.full(mask.shape, -1, dtype=np.int32)
    if not np.any(mask):
        return labels, []

    if mode == 'components':
        n, cc = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        labels = cc.astype(np.int32) - 1
    else:
        feats = _cluster_features(img, depth, mask, mode, depth_weight, color_weight)
        labels[mask] = _kmeans(feats, k, seed=seed)
        if mode == 'lineart_watershed':
            labels = _watershed(img, mask, labels)
        elif mode == 'color':
            # colour clusters are noisy, clean them up a little
            labels = _fill_unlabeled(np.where(_open_labels(labels, mask), labels, -1), mask)
        if split_components:
            labels = _split_components(labels, mask)

    if max_pieces == 0:
        # disconnected bits of one cluster become their own pieces, but never more than k in total
        max_pieces = k
    labels = _merge_small(labels, mask, min_area=min_area_ratio * mask.sum(), max_pieces=max_pieces)

    ids = np.unique(labels[mask])
    meds = np.array([np.median(depth[labels == i]) for i in ids])
    if mode == 'components' or np.ptp(meds) < 1e-4:
        # no usable depth: bigger pieces go behind smaller ones (buttons on a jacket, etc.)
        areas = np.array([(labels == i).sum() for i in ids])
        order = list(ids[np.lexsort((areas, meds))]) if np.ptp(meds) >= 1e-4 else list(ids[np.argsort(areas)])
    else:
        order = list(ids[np.argsort(meds)])
    return labels, order


def _open_labels(labels, mask, ksize=1):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1))
    keep = np.zeros(mask.shape, dtype=bool)
    for lid in np.unique(labels[mask]):
        keep |= cv2.morphologyEx((labels == lid).astype(np.uint8), cv2.MORPH_OPEN, element) > 0
    return keep & mask


def _get_inpaint_fn(inpaint):
    if inpaint == 'lama':
        import torch
        from annotators.lama_inpainter import apply_inpaint
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return lambda rgb, m: apply_inpaint(rgb, m, device=device)
    if inpaint == 'cv2':
        return lambda rgb, m: cv2.inpaint(rgb, m, 5, cv2.INPAINT_TELEA)
    if inpaint == 'none':
        return None
    raise ValueError(f'invalid inpaint method {inpaint}')


def split_part_by_labels(img, depth, labels, order, inpaint='cv2', max_extend=None, feather_sigma=0.7):
    """Cut img into one piece per label, front (order[0]) to back.

    The area a piece hides is filled into the pieces behind it: hidden pixels go to the
    nearest piece behind, within max_extend pixels (default: ~12% of the layer size), and
    their colour is inpainted from what is visible of the remaining pieces.
    """
    h, w = labels.shape
    alpha = img[..., 3]
    rgb = img[..., :3].copy()
    mask = labels >= 0
    if max_extend is None:
        max_extend = max(8, int(0.12 * max(h, w)))
    inpaint_fn = _get_inpaint_fn(inpaint)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    pieces = []
    front = np.zeros((h, w), dtype=bool)
    for rank, lid in enumerate(order):
        own = labels == lid
        piece_mask = own.copy()
        if rank > 0 and inpaint_fn is not None:
            # hidden region: pixels of pieces in front, assigned to the nearest piece behind them
            behind = mask & ~front
            rest = np.where(behind, labels, -1)
            filled = _fill_unlabeled(rest, mask)
            dist = cv2.distanceTransform((~own).astype(np.uint8), cv2.DIST_L2, 5)
            piece_mask |= front & (filled == lid) & (dist <= max_extend)
        hidden = piece_mask & ~own

        prgb = rgb
        if np.any(hidden):
            # inpaint from what is still visible behind the pieces already cut out
            src = rgb.copy()
            src[front] = 0
            fm = (front | hidden).astype(np.uint8) * 255
            fm = cv2.dilate(fm, element)
            fm[own & ~hidden] = 0
            prgb = inpaint_fn(np.ascontiguousarray(src), fm)
            prgb = np.where(own[..., None], rgb, prgb).astype(np.uint8)

        a = piece_mask.astype(np.float32)
        if rank < len(order) - 1 and feather_sigma > 0:
            # soften the internal cut edge, the piece behind covers what shows through
            a = cv2.GaussianBlur(a, (3, 3), sigmaX=feather_sigma, sigmaY=feather_sigma)
            a = np.maximum(a, own.astype(np.float32))
        palpha = np.round(alpha.astype(np.float32) * a).astype(np.uint8)
        # hidden pixels behind an anti-aliased front edge are still fully covered by this piece
        palpha[hidden] = np.maximum(palpha[hidden], 255 * (alpha[hidden] > ALPHA_THR))

        pdepth = depth.astype(np.float32).copy()
        dmed = float(np.median(depth[own]))
        # 8-bit depth gives ties; keep the cut order strict so a piece never lands above the ones hiding it
        if pieces and dmed <= pieces[-1]['depth_median']:
            dmed = pieces[-1]['depth_median'] + 1e-5
        pdepth[hidden] = dmed
        pieces.append({
            'img': np.concatenate([prgb, palpha[..., None]], axis=-1),
            'depth': pdepth,
            'depth_median': dmed,
        })
        front |= own
    return pieces


def split_part(part, mode='lineart_watershed', k=4, inpaint='cv2', seed=0, **kwargs):
    """Split one part dict into a list of part dicts (front to back, cropped to their content)."""
    img = part['img']
    depth = part['depth']
    if depth.dtype == np.uint8:
        depth = depth.astype(np.float32) / 255.
    label_kwargs = {k_: kwargs[k_] for k_ in _LABEL_KEYS if k_ in kwargs}
    labels, order = compute_labels(img, depth, mode=mode, k=k, seed=seed, **label_kwargs)
    if len(order) <= 1:
        return [part]
    pieces = split_part_by_labels(img, depth, labels, order, inpaint=inpaint,
                                  max_extend=kwargs.get('max_extend'))
    x1, y1 = part['xyxy'][:2]
    rst = []
    for p in pieces:
        m = p['img'][..., 3] > 0
        if not np.any(m):
            continue
        bx, by, bw, bh = cv2.boundingRect(m.astype(np.uint8))
        rst.append({
            'img': p['img'][by: by + bh, bx: bx + bw],
            'depth': p['depth'][by: by + bh, bx: bx + bw],
            'depth_median': p['depth_median'],
            'xyxy': [int(x1 + bx), int(y1 + by), int(x1 + bx + bw), int(y1 + by + bh)],
        })
    return rst


def _part_area(part):
    return int((part['img'][..., 3] > ALPHA_THR).sum())


def _allocate_k(rules, tag2pinfo, target_layers):
    """Give each weighted rule a piece count so the total layer count lands near target_layers."""
    weighted = [r for r in rules if r['_auto']]
    if target_layers is None or not weighted:
        for r in weighted:
            r['k'] = r.get('min_k', 2)
        return
    fixed_extra = sum(r['k'] - 1 for r in rules if not r['_auto'])
    budget = target_layers - len(tag2pinfo) - fixed_extra
    scores = np.array([r.get('weight', 1.0) * np.sqrt(r['_area']) for r in weighted], dtype=np.float64)
    if budget <= 0 or scores.sum() == 0:
        for r in weighted:
            r['k'] = r.get('min_k', 1)
        return
    lo = np.array([r.get('min_k', 1) for r in weighted])
    hi = np.array([max(r.get('max_k', 40), r.get('min_k', 1)) for r in weighted])
    # k pieces add k - 1 layers; hand out the budget by score, refilling rules that hit max_k
    ks = lo.copy()
    left = budget - (ks - 1).sum()
    while left > 0:
        room = ks < hi
        if not np.any(room):
            break
        sc = np.where(room, scores, 0)
        if sc.sum() == 0:
            sc = room.astype(np.float64)
        share = sc / sc.sum() * left
        add = np.minimum(np.floor(share).astype(int), hi - ks)
        if add.sum() == 0:
            # largest remainders get the last few pieces
            for i in np.argsort(-share):
                if left == 0:
                    break
                if ks[i] < hi[i]:
                    ks[i] += 1
                    left -= 1
            break
        ks += add
        left -= add.sum()
    for r, kk in zip(weighted, ks):
        r['k'] = int(kk)


def apply_fine_split(tag2pinfo, config, target_layers=None, inpaint=None, seed=0, verbose=True):
    """Split parts in tag2pinfo in place according to config rules.

    config = {
        "inpaint": "cv2",                      # cv2 | lama | none
        "target_layers": 70,                   # optional, overridden by the argument
        "rules": [
            {"tags": ["front hair"], "mode": "lineart_watershed", "weight": 4},
            {"tags": ["legwear"], "mode": "components", "k": 2, "lr": true},
            ...
        ]
    }
    A rule with "k" uses that piece count; otherwise pieces are allocated from the
    target_layers budget by "weight" x sqrt(area). Several tags in one rule are each split
    separately. Pieces are named <tag>-0 ... <tag>-N front to back; with "lr": true a
    two-piece split is named <tag>-r / <tag>-l instead (left in the image = character's right).
    """
    config = load_fine_split_config(config)
    if target_layers is None:
        target_layers = config.get('target_layers')
    if inpaint is None:
        inpaint = config.get('inpaint', 'cv2')

    # expand rules to one entry per present tag
    rules = []
    for rule in config.get('rules', []):
        for tag in rule['tags']:
            if tag in tag2pinfo and _part_area(tag2pinfo[tag]) > 0:
                r = {k: v for k, v in rule.items() if k != 'tags'}
                r['tag'] = tag
                r['_auto'] = 'k' not in rule
                r['_area'] = _part_area(tag2pinfo[tag])
                rules.append(r)

    n_before = len(tag2pinfo)
    for attempt in range(3):
        _allocate_k(rules, tag2pinfo, target_layers)
        n_pieces = {}
        for r in rules:
            if r['k'] <= 1:
                continue
            part = tag2pinfo[r['tag']]
            depth = part['depth'].astype(np.float32) / 255. if part['depth'].dtype == np.uint8 else part['depth']
            _, order = compute_labels(
                part['img'], depth, mode=r.get('mode', 'lineart_watershed'), k=r['k'], seed=seed,
                **{k: v for k, v in r.items() if k in _LABEL_KEYS})
            n_pieces[r['tag']] = max(len(order), 1)
        total = n_before + sum(v - 1 for v in n_pieces.values())
        if target_layers is None or total >= target_layers or attempt == 2:
            break
        # small pieces were merged away: ask the weighted rules for more
        short = target_layers - total
        for r in sorted([r for r in rules if r['_auto']], key=lambda r: -r['_area'])[:short]:
            r['min_k'] = r['k'] + 1

    # final pass with inpainting (the label passes above skip it to stay fast)
    n_split = 0
    for r in rules:
        if r['k'] <= 1:
            continue
        tag = r['tag']
        part = tag2pinfo[tag]
        pieces = split_part(
            part, mode=r.get('mode', 'lineart_watershed'), k=r['k'], inpaint=inpaint, seed=seed,
            **{k: v for k, v in r.items() if k in _LABEL_KEYS + ('max_extend',)})
        if len(pieces) <= 1:
            continue
        names = None
        if r.get('lr') and len(pieces) == 2:
            # same convention as part_lr_split: left in the image is the character's right
            pieces.sort(key=lambda p: p['xyxy'][0] + p['xyxy'][2])
            names = [f'{tag}-r', f'{tag}-l']
        _insert_pieces(tag2pinfo, tag, pieces, names)
        n_split += 1
        if verbose:
            print(f'[fine_split] {tag}: {len(pieces)} pieces ({r.get("mode", "lineart_watershed")}, k={r["k"]})')

    if verbose:
        print(f'[fine_split] {n_before} -> {len(tag2pinfo)} layers'
              + (f' (target {target_layers})' if target_layers else ''))
    return tag2pinfo


def _insert_pieces(tag2pinfo, tag, pieces, names=None):
    """Replace tag with its pieces, keeping the dict order and the source depth order."""
    src = tag2pinfo[tag]
    src_med = src.get('depth_median')
    items = list(tag2pinfo.items())
    idx = [k for k, _ in items].index(tag)
    new_items = []
    for i, p in enumerate(pieces):
        name = names[i] if names is not None else f'{tag}-{i}'
        if src_med is not None and not np.isfinite(p['depth_median']):
            p['depth_median'] = src_med
        p['tag'] = name
        new_items.append((name, p))
    if src.get('_fixed_order'):
        # depth is unknown (plain PSD input): keep pieces at the source slot, front piece on top
        eps = src['_order_eps'] / (len(pieces) + 1)
        ranks = np.argsort(np.argsort([p['depth_median'] for _, p in new_items]))
        for rank, (_, p) in zip(ranks, new_items):
            p['depth_median'] = src_med - (len(pieces) - 1 - rank) * eps
            p['_fixed_order'] = True
            p['_order_eps'] = eps
    for _, p in new_items:
        # save_part() recomputes depth_median from 8-bit depth; further_extr restores this value
        p['fine_split_depth_median'] = p['depth_median']
    tag2pinfo.clear()
    tag2pinfo.update(items[:idx] + new_items + items[idx + 1:])


def load_psd_parts(psdp):
    """Read a See-through PSD (or any flat layered PSD) into part dicts.

    Uses the sidecar <psd>.json and <name>_depth.psd written by See-through when present;
    otherwise layer order stands in for depth (bottom layer = farthest).
    """
    from psd_tools import PSDImage
    psd = PSDImage.open(psdp)
    w, h = psd.width, psd.height
    layers = [l for l in psd.descendants() if not l.is_group() and l.width > 0 and l.height > 0]

    depth_layers = {}
    depthp = osp.splitext(psdp)[0] + '_depth.psd'
    if osp.exists(depthp):
        for l in PSDImage.open(depthp).descendants():
            if not l.is_group() and l.width > 0:
                depth_layers[l.name] = np.array(l.topil().convert('L'), dtype=np.float32) / 255.

    n = len(layers)
    tag2pinfo = {}
    for i, l in enumerate(layers):
        img = np.array(l.topil().convert('RGBA'))
        x1, y1 = l.left, l.top
        name = l.name
        while name in tag2pinfo:
            name += '_'
        m = img[..., 3] > ALPHA_THR
        if name in depth_layers and depth_layers[name].shape == img.shape[:2]:
            depth = depth_layers[name]
            dmed = float(np.median(depth[m])) if np.any(m) else 1.
            fixed = False
        else:
            # first layer in the PSD is the bottom one
            dmed = 1. - i / max(n, 1)
            depth = np.full(img.shape[:2], dmed, dtype=np.float32)
            fixed = True
        tag2pinfo[name] = {'img': img, 'depth': depth, 'depth_median': dmed, 'tag': name,
                           'xyxy': [x1, y1, x1 + img.shape[1], y1 + img.shape[0]]}
        if fixed:
            tag2pinfo[name]['_fixed_order'] = True
            tag2pinfo[name]['_order_eps'] = 1. / max(n, 1)
    return tag2pinfo, (h, w)


def save_parts_psd(tag2pinfo, frame_size, savep):
    """Write <savep> and <savep>_depth.psd, layers sorted far to near."""
    parts = []
    for tag, p in tag2pinfo.items():
        d = p['depth']
        if d.dtype != np.uint8:
            d = np.round(np.clip(d, 0, 1) * 255).astype(np.uint8)
        parts.append({'img': p['img'], 'depth': d, 'xyxy': [int(v) for v in p['xyxy']],
                      'tag': tag, 'depth_median': float(p['depth_median'])})
    parts.sort(key=lambda x: x['depth_median'], reverse=True)
    h, w = frame_size
    # utils.io_utils.save_psd swaps width/height, which only works for square canvases
    _write_psd(savep, parts, h, w, 'RGBA', 'img')
    _write_psd(osp.splitext(savep)[0] + '_depth.psd', parts, h, w, 'L', 'depth')
    return parts


def _write_psd(savep, parts, h, w, mode, img_key):
    from PIL import Image
    from psd_tools import PSDImage
    psd = PSDImage.new(mode=mode, size=(w, h), depth=8)
    for p in parts:
        x1, y1 = p['xyxy'][:2]
        psd.create_pixel_layer(Image.fromarray(p[img_key]), name=p['tag'], top=int(y1), left=int(x1), opacity=255)
    psd.save(savep)
