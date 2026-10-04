"""Synthetic full-body front-facing character in See-through v3 output format.

Produces a directory identical in layout to what ``apply_layerdiff`` + ``apply_marigold``
write (``<tag>.png`` full-canvas RGBA at inference resolution, ``<tag>_depth.png``,
``src_img.png``, ``info.json``) plus test-only extras:
``original.png`` (non-square original image) and ``gt.json`` (ground-truth joints).

Hidden regions are "inpainted" (e.g. the arm continues under the sleeve, legs under
the skirt, back hair behind the head) so tests can verify that hidden pixels survive.
Character-right side is on the SCREEN LEFT (front-facing character).
"""
import os
import os.path as osp
import json
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 'common'))
from utils.cv import center_square_pad_resize, img_alpha_blending  # noqa: E402

V3_TAGS = ['front hair', 'back hair', 'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear',
           'footwear', 'tail', 'wings', 'objects', 'headwear', 'face', 'irides', 'eyebrow', 'eyewhite',
           'eyelash', 'eyewear', 'ears', 'earwear', 'nose', 'mouth']

# larger depth = farther (See-through convention: sorted reverse=True -> bottom first)
DEPTH = {'back hair': 235, 'legwear': 190, 'footwear': 185, 'neck': 170, 'handwear': 160, 'bottomwear': 150,
         'topwear': 140, 'ears': 128, 'face': 125, 'neckwear': 118, 'eyewhite': 112, 'irides': 108,
         'nose': 108, 'mouth': 108, 'eyelash': 104, 'eyebrow': 102, 'front hair': 90, 'headwear': 80}

COLORS = {'back hair': (90, 60, 160), 'front hair': (120, 80, 200), 'face': (250, 220, 200), 'ears': (240, 205, 185),
          'neck': (235, 200, 180), 'eyewhite': (255, 255, 255), 'irides': (40, 120, 220), 'eyelash': (30, 20, 40),
          'eyebrow': (70, 40, 110), 'nose': (230, 180, 170), 'mouth': (200, 80, 90), 'topwear': (240, 240, 250),
          'handwear': (245, 210, 190), 'bottomwear': (40, 50, 110), 'legwear': (20, 20, 30),
          'footwear': (110, 60, 30), 'headwear': (230, 60, 80), 'neckwear': (220, 30, 50)}


def _line(m, p0, p1, t):
    cv2.line(m, tuple(map(int, p0)), tuple(map(int, p1)), 255, int(t), lineType=cv2.LINE_8)
    cv2.circle(m, tuple(map(int, p0)), int(t // 2), 255, -1)
    cv2.circle(m, tuple(map(int, p1)), int(t // 2), 255, -1)


def build_masks(W=768, H=1024):
    """Masks in ORIGINAL image coordinates. Returns dict tag->uint8 mask, gt joints."""
    s = W / 768.
    S = lambda *v: [int(round(x * s)) for x in v]  # noqa: E731
    cx = W // 2
    z = lambda: np.zeros((H, W), np.uint8)  # noqa: E731
    m = {}
    gt = {}

    # --- head ---
    face = z(); cv2.ellipse(face, (cx, S(210)[0]), tuple(S(78, 92)), 0, 0, 360, 255, -1); m['face'] = face
    ears = z()
    for sx in (-1, 1):
        cv2.ellipse(ears, (cx + sx * S(80)[0], S(215)[0]), tuple(S(14, 26)), 0, 0, 360, 255, -1)
    m['ears'] = ears
    back = z()
    cv2.ellipse(back, (cx, S(220)[0]), tuple(S(125, 150)), 0, 0, 360, 255, -1)
    pts = np.array([S(cx / s - 125, 220), S(cx / s + 125, 220), S(cx / s + 140, 520), S(cx / s - 140, 520)], np.int32)
    cv2.fillPoly(back, [pts], 255)
    m['back hair'] = back
    front = z()
    gt['front_hair_tips'] = []
    cv2.ellipse(front, (cx, S(200)[0]), tuple(S(100, 115)), 0, 180, 360, 255, -1)  # top cap
    for i, dx in enumerate([-70, -40, -10, 20, 50, 75]):
        x0 = cx + S(dx)[0]
        tri = np.array([[x0 - S(22)[0], S(150)[0]], [x0 + S(22)[0], S(150)[0]], [x0 + S(4)[0], S(185 + (i % 2) * 12)[0]]], np.int32)
        cv2.fillPoly(front, [tri], 255)
        gt['front_hair_tips'].append([x0 + S(4)[0], S(185 + (i % 2) * 12)[0] - S(8)[0]])
    for sx in (-1, 1):  # side locks
        x0 = cx + sx * S(92)[0]
        poly = np.array([[x0 - S(16)[0], S(180)[0]], [x0 + S(16)[0], S(180)[0]], [x0 + sx * S(6)[0], S(360)[0]]], np.int32)
        cv2.fillPoly(front, [poly], 255)
        cv2.rectangle(front, (x0 - S(16)[0], S(150)[0]), (x0 + S(16)[0], S(185)[0]), 255, -1)
    m['front hair'] = front
    for tag, ys, ax in [('eyewhite', 215, (18, 12)), ('irides', 217, (9, 11))]:
        mm = z()
        for sx in (-1, 1):
            cv2.ellipse(mm, (cx + sx * S(34)[0], S(ys)[0]), tuple(S(*ax)), 0, 0, 360, 255, -1)
        m[tag] = mm
    for tag, y, t in [('eyelash', 202, 5), ('eyebrow', 185, 4)]:
        mm = z()
        for sx in (-1, 1):
            _line(mm, (cx + sx * S(16)[0], S(y)[0]), (cx + sx * S(54)[0], S(y + 2)[0]), max(2, S(t)[0]))
        m[tag] = mm
    nose = z(); cv2.circle(nose, (cx, S(240)[0]), max(2, S(3)[0]), 255, -1); m['nose'] = nose
    mouth = z(); _line(mouth, (cx - S(12)[0], S(265)[0]), (cx + S(12)[0], S(265)[0]), max(2, S(4)[0])); m['mouth'] = mouth

    # --- body ---
    neck = z(); cv2.rectangle(neck, (cx - S(22)[0], S(280)[0]), (cx + S(22)[0], S(350)[0]), 255, -1); m['neck'] = neck
    # arms: character RIGHT arm is on SCREEN LEFT (sx=-1)
    arms = z(); sleeves = z()
    for side, sx in (('R', -1), ('L', 1)):
        sh = (cx + sx * S(78)[0], S(350)[0])
        el = (cx + sx * S(128)[0], S(490)[0])
        wr = (cx + sx * S(150)[0], S(610)[0])
        hd = (cx + sx * S(155)[0], S(645)[0])
        gt[f'arm_{side}'] = {'shoulder': sh, 'elbow': el, 'wrist': wr, 'hand': hd}
        _line(arms, sh, el, S(40)[0]); _line(arms, el, wr, S(34)[0])
        cv2.ellipse(arms, hd, tuple(S(22, 34)), 0, 0, 360, 255, -1)
        _line(sleeves, sh, el, S(56)[0])
        el2 = el; wr_s = (int(el[0] + (wr[0] - el[0]) * 0.85), int(el[1] + (wr[1] - el[1]) * 0.85))
        _line(sleeves, el2, wr_s, S(50)[0])
    m['handwear'] = arms
    torso = z()
    pts = np.array([[cx - S(85)[0], S(335)[0]], [cx + S(85)[0], S(335)[0]], [cx + S(70)[0], S(540)[0]], [cx - S(70)[0], S(540)[0]]], np.int32)
    cv2.fillPoly(torso, [pts], 255)
    m['topwear'] = np.maximum(torso, sleeves)
    skirt = z()
    pts = np.array([[cx - S(75)[0], S(520)[0]], [cx + S(75)[0], S(520)[0]], [cx + S(125)[0], S(690)[0]], [cx - S(125)[0], S(690)[0]]], np.int32)
    cv2.fillPoly(skirt, [pts], 255)
    m['bottomwear'] = skirt
    legs = z(); shoes = z()
    for side, sx in (('R', -1), ('L', 1)):
        hip = (cx + sx * S(40)[0], S(560)[0])
        kn = (cx + sx * S(44)[0], S(770)[0])
        an = (cx + sx * S(46)[0], S(940)[0])
        gt[f'leg_{side}'] = {'hip': hip, 'knee': kn, 'ankle': an}
        _line(legs, hip, kn, S(50)[0]); _line(legs, kn, an, S(40)[0])
        cv2.ellipse(shoes, (an[0] + sx * S(6)[0], S(968)[0]), tuple(S(30, 22)), 0, 0, 360, 255, -1)
    m['legwear'] = legs
    m['footwear'] = shoes
    # accessories: hair ornament (2 separate pieces) + neck ribbon (knot + 2 tails, connected)
    hw = z()
    cv2.circle(hw, (cx + S(70)[0], S(120)[0]), S(16)[0], 255, -1)
    cv2.circle(hw, (cx - S(80)[0], S(135)[0]), S(10)[0], 255, -1)
    m['headwear'] = hw
    nw = z()
    cv2.circle(nw, (cx, S(345)[0]), S(12)[0], 255, -1)
    for sx in (-1, 1):
        tri = np.array([[cx, S(345)[0]], [cx + sx * S(45)[0], S(325)[0]], [cx + sx * S(45)[0], S(365)[0]]], np.int32)
        cv2.fillPoly(nw, [tri], 255)
        _line(nw, (cx, S(350)[0]), (cx + sx * S(25)[0], S(420)[0]), S(12)[0])
    m['neckwear'] = nw
    return m, gt


def _texture(H, W, color, seed):
    rng = np.random.default_rng(seed)
    base = np.array(color, np.float32)[None, None]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    shade = (np.sin(xx / 23. + seed) * np.cos(yy / 31.) * 10)[..., None]
    noise = rng.normal(0, 3, (H, W, 1)).astype(np.float32)
    return np.clip(base + shade + noise, 0, 255).astype(np.uint8)


def make_sample(out_dir, W=768, H=1024, resolution=1024, seed=0, srcname='synth'):
    """Write a See-through-style output directory ``out_dir/srcname``. Returns its path."""
    masks, gt = build_masks(W, H)
    saved = osp.join(out_dir, srcname)
    os.makedirs(saved, exist_ok=True)
    layers_orig = {}
    for i, (tag, mk) in enumerate(masks.items()):
        alpha = cv2.GaussianBlur(mk, (3, 3), 0.8)
        rgb = _texture(H, W, COLORS[tag], seed + i)
        layers_orig[tag] = np.concatenate([rgb, alpha[..., None]], -1)
    order = sorted(layers_orig, key=lambda t: DEPTH[t], reverse=True)
    original = img_alpha_blending([layers_orig[t] for t in order], premultiplied=False)
    # the "true" original has extra fine detail the generated layers don't reproduce exactly
    rng = np.random.default_rng(seed + 999)
    detail = rng.integers(-6, 7, original[..., :3].shape)
    original[..., :3] = np.clip(original[..., :3].astype(np.int32) + detail, 0, 255).astype(np.uint8)
    Image.fromarray(original).save(osp.join(saved, 'original.png'))

    fullpage, pad_size, pad_pos = center_square_pad_resize(original, resolution, return_pad_info=True)
    Image.fromarray(fullpage).save(osp.join(saved, 'src_img.png'))
    info = {'parts': {}}
    for tag in V3_TAGS:
        if tag in layers_orig:
            lay = center_square_pad_resize(layers_orig[tag], resolution)
            d = np.full(lay.shape[:2], 255, np.uint8)
            msk = lay[..., -1] > 15
            d[msk] = DEPTH[tag]
        else:
            lay = np.zeros((resolution, resolution, 4), np.uint8)
            d = np.full(lay.shape[:2], 255, np.uint8)
        Image.fromarray(lay).save(osp.join(saved, f'{tag}.png'))
        Image.fromarray(d).save(osp.join(saved, f'{tag}_depth.png'))
        info['parts'][tag] = {}
    with open(osp.join(saved, 'info.json'), 'w') as f:
        json.dump(info, f)
    with open(osp.join(saved, 'gt.json'), 'w') as f:
        json.dump({'joints_original': gt, 'original_size': [H, W], 'resolution': resolution,
                   'pad_size': list(pad_size), 'pad_pos': list(pad_pos)}, f, indent=1)
    return saved


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/synth_seethrough'
    print(make_sample(out))
