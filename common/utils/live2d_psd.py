"""PSD export for Live2D detailed parts (full-canvas layers, offset 0, optional groups).

Uses psd-tools (already a See-through dependency).  Z-order is the part order given
(first = bottom).  Groups (Hair / Face / Body / Clothes / Accessory / Other) are created from
CONSECUTIVE runs of the same group so that the drawing order is never changed; a group that
appears again higher up the stack gets a numbered folder (e.g. "Hair_2" for front hair above the face).
If group creation fails, layers are written flat with a "<Group>_" name prefix.
"""
from __future__ import annotations

import json
from typing import List, Sequence

import numpy as np
from PIL import Image


def _runs(parts):
    runs = []
    for p in parts:
        if runs and runs[-1][0] == p.group:
            runs[-1][1].append(p)
        else:
            runs.append((p.group, [p]))
    return runs


def save_live2d_psd(savep: str, parts: Sequence, size_hw, use_groups: bool = True) -> dict:
    """Write ``parts`` (bottom -> top) as a PSD of canvas ``size_hw`` = (H, W). Returns layout info."""
    from psd_tools import PSDImage
    H, W = size_hw
    for p in parts:
        if p.img.shape[:2] != (H, W):
            raise ValueError(f'{p.name}: {p.img.shape[:2]} != canvas {(H, W)}')
    mode = 'groups' if use_groups else 'prefix'
    if use_groups:
        try:
            psd = PSDImage.new(mode='RGBA', size=(W, H), depth=8)  # PIL order: (width, height)
            seen = {}
            for g, plist in _runs(parts):
                seen[g] = seen.get(g, 0) + 1
                gname = g if seen[g] == 1 else f'{g}_{seen[g]}'
                layers = [psd.create_pixel_layer(Image.fromarray(p.img), name=p.name, top=0, left=0) for p in plist]
                psd.create_group(layers, name=gname)
            psd.save(savep)
        except Exception as e:  # noqa: BLE001
            print(f'[live2d] PSD groups failed ({type(e).__name__}: {e}); writing prefixed flat layers')
            mode = 'prefix'
    if mode == 'prefix':
        psd = PSDImage.new(mode='RGBA', size=(W, H), depth=8)
        for p in parts:
            psd.create_pixel_layer(Image.fromarray(p.img), name=f'{p.group}_{p.name}', top=0, left=0)
        psd.save(savep)
    meta = {'canvas_hw': [H, W], 'mode': mode, 'layers': []}
    for i, p in enumerate(parts):
        ys, xs = np.nonzero(p.img[..., 3] > 0)
        bb = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
        meta['layers'].append({'order': i, 'name': p.name, 'group': p.group, 'side': p.side, 'source': p.source,
                               'method': p.method, 'depth': float(p.depth), 'bbox_xyxy': bb})
    with open(savep + '.json', 'w') as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    return meta


def read_psd_layers(psdp: str) -> List[dict]:
    """Read back every pixel layer as a full-canvas RGBA (for tests): [{name, group, img}] bottom->top."""
    from psd_tools import PSDImage
    psd = PSDImage.open(psdp)
    W, H = psd.size
    out = []

    def walk(container, group):
        for l in container:
            if l.is_group():
                walk(l, l.name)
            else:
                full = Image.new('RGBA', (W, H), (0, 0, 0, 0))
                full.paste(l.topil().convert('RGBA'), (l.left, l.top))
                out.append({'name': l.name, 'group': group, 'img': np.array(full), 'offset': (l.left, l.top),
                            'size': l.size})
    walk(psd, None)
    return out
