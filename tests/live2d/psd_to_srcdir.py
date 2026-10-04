"""Convert a normal-mode See-through PSD (e.g. fetched from the HF Space) back into a
See-through output directory that the detailed split can consume.

    python tests/live2d/psd_to_srcdir.py FETCH_DIR [OUT_SRCDIR]

FETCH_DIR is what fetch_hf_space.py wrote (normal.psd, input.png, gallery/).
Writes OUT_SRCDIR (default FETCH_DIR/srcd) with:
  src_img.png                  = center_square_pad_resize(input, psd canvas size)  (same as apply_layerdiff)
  <tag>.png                    = PSD layer pasted at its offset on a full transparent canvas (lossless)
  <tag>_depth.png              = constant depth from PSD stacking order (PSD has no depth maps)
  info.json                    = {'parts': {tag: {}}}
Tags absent from the PSD but non-empty in the gallery (lossy webp) are added and listed in
``info.json['from_gallery']``.
Limitation: the normal PSD was produced by further_extr, so nose/mouth RGB already come from
src_img and alpha<=10 pixels outside a layer's bbox are gone; otherwise layers are the exact
inpainted RGBA produced by the model.
"""
import json
import os
import os.path as osp
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 'common'))
from utils.cv import center_square_pad_resize  # noqa: E402


def convert(fetch_dir, out=None):
    from psd_tools import PSDImage
    out = out or osp.join(fetch_dir, 'srcd')
    os.makedirs(out, exist_ok=True)
    psd = PSDImage.open(osp.join(fetch_dir, 'normal.psd'))
    W, H = psd.size
    inp = np.array(Image.open(osp.join(fetch_dir, 'input.png')).convert('RGBA'))
    Image.fromarray(center_square_pad_resize(inp, W)).save(osp.join(out, 'src_img.png'))
    layers = list(psd)
    n = len(layers)
    info = {'parts': {}, 'from_gallery': []}
    for i, l in enumerate(layers):  # first = bottom = farthest
        full = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        full.paste(l.topil().convert('RGBA'), l.offset)
        full.save(osp.join(out, f'{l.name}.png'))
        d = int(round(250 - 200 * i / max(n - 1, 1)))
        Image.fromarray(np.full((H, W), d, np.uint8)).save(osp.join(out, f'{l.name}_depth.png'))
        info['parts'][l.name] = {}
    gd = osp.join(fetch_dir, 'gallery')
    if osp.isdir(gd):
        for f in sorted(os.listdir(gd)):
            tag = osp.splitext(f)[0]
            if tag in info['parts'] or tag in ('head', 'src_img', 'src_head', 'reconstruction'):
                continue
            img = np.array(Image.open(osp.join(gd, f)).convert('RGBA'))
            if (img[..., 3] > 10).sum() == 0:
                continue
            Image.fromarray(img).save(osp.join(out, f'{tag}.png'))
            Image.fromarray(np.full((H, W), 128, np.uint8)).save(osp.join(out, f'{tag}_depth.png'))
            info['parts'][tag] = {}
            info['from_gallery'].append(tag)
    json.dump(info, open(osp.join(out, 'info.json'), 'w'), indent=1)
    return out


if __name__ == '__main__':
    print(convert(*sys.argv[1:]))
