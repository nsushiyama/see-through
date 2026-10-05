"""Background-leakage cleanup (legwear predicted as an opaque sheet over the background) and
row-wise left/right split of legs that touch (knees together)."""
import os.path as osp
import shutil

import numpy as np
from PIL import Image

import utils.live2d_split as L


def _leaky_copy(synth_dir, tmp_path, bg_rgb=(255, 255, 255)):
    """Copy synth sample and paint a grey opaque sheet over the INPUT background only
    (matching the real HF Space failure mode: legwear covering white page, not the character)."""
    d = str(tmp_path / 'leaky')
    shutil.copytree(synth_dir, d)
    src = np.array(Image.open(osp.join(d, 'src_img.png')).convert('RGBA')).astype(np.float32)
    H, W = src.shape[:2]
    a = src[..., 3:] / 255.
    rgb = src[..., :3] * a + np.array(bg_rgb, np.float32) * (1 - a)
    rng = np.random.default_rng(0)
    rgb = np.clip(rgb + rng.integers(-2, 3, rgb.shape), 0, 255)
    out = np.concatenate([rgb, np.full((H, W, 1), 255.)], -1).astype(np.uint8)
    out[:, : W // 16, 3] = 0          # some padding
    out[:, : W // 16, :3] = 0
    out[-1, W // 16:, :3] = 120       # a 1-px frame line at the bottom (like real inputs)
    Image.fromarray(out).save(osp.join(d, 'src_img.png'))
    bg = L.background_mask(out)
    lw = np.array(Image.open(osp.join(d, 'legwear.png')).convert('RGBA'))
    good = lw[..., 3] > 10
    leak = bg & ~good                 # only over visible background (not over the character)
    lw[leak] = (200, 198, 199, 255)
    Image.fromarray(lw).save(osp.join(d, 'legwear.png'))
    return d, good, out, leak


def test_background_mask_and_cleanup(synth_dir, tmp_path):
    d, good, page, leak = _leaky_copy(synth_dir, tmp_path)
    fp, parts = L.load_semantic_layers(d)
    bg = L.background_mask(fp)
    char = np.array(Image.open(osp.join(synth_dir, 'src_img.png')))[..., 3] > 10
    assert not np.any(bg & L.cv2.erode(char.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
    assert bg.mean() > 0.5
    leg = next(p for p in parts if p.source == 'legwear')
    assert L.is_suspicious(leg)
    rep = L.SplitReport()
    ctx = L.SplitContext(canvas_hw=fp.shape[:2], fullpage=fp)
    out = L.stage_bg_cleanup(parts, rep, ctx)
    leg2 = next(p for p in out if p.source == 'legwear')
    assert not L.is_suspicious(leg2) and leg2.method == 'bg_cleanup'
    assert (leg2.mask & good).sum() >= 0.97 * good.sum()       # real legs (incl. hidden parts) kept
    # eroded bg margin leaves a thin ring; far-from-character leakage must be gone
    far = L.cv2.erode(leak.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    assert (leg2.mask & far).sum() <= 0.01 * good.sum()
    assert (leg2.mask & leak).sum() <= 0.08 * good.sum()
    assert not leg2.mask[-3:].any()                             # frame line removed
    assert any('bg cleanup' in e for e in rep.events)
    for a, b in zip(parts, out):
        if a.source != 'legwear':
            assert a is b


def test_pipeline_splits_legs_after_cleanup(synth_dir, tmp_path):
    d, _, _, _ = _leaky_copy(synth_dir, tmp_path)
    res = L.run_detailed_split(d, out_psd=str(tmp_path / 'o.psd'), restore_visible=False)
    for n in ['thigh_R', 'thigh_L', 'lower_leg_R', 'lower_leg_L']:
        assert n in res['names'], res['names']


def test_no_cleanup_without_leak(synth_dir):
    fp, parts = L.load_semantic_layers(synth_dir)
    out = L.stage_bg_cleanup(parts, L.SplitReport(), L.SplitContext(canvas_hw=fp.shape[:2], fullpage=fp))
    assert all(a is b for a, b in zip(parts, out))


def _legs_part(touch=True, blob=False):
    H, W = 400, 300
    m = np.zeros((H, W), np.uint8)
    L.cv2.line(m, (120, 20), (140, 380), 1, 34)    # character right leg (screen left)
    L.cv2.line(m, (180, 20), (160, 380), 1, 34)    # character left leg
    if touch:
        m[180:220, 120:180] = 1                    # knees pressed together
    if blob:
        m[2:8, 200:215] = 1                        # stray blob above the left leg
    img = np.zeros((H, W, 4), np.uint8)
    img[..., :3] = 180
    img[..., 3] = m * 255
    return L.Part('leg', img, 0.5, 'legwear', 'Body')


def test_split_lr_rows_touching_legs():
    p = _legs_part()
    assert len(L.components(p.mask)) == 1
    r, l = L.split_lr_rows(p, cx=150.0)
    assert r.side == 'character_right' and l.side == 'character_left'
    assert np.nonzero(r.mask)[1].mean() < np.nonzero(l.mask)[1].mean()
    assert np.array_equal(r.mask | l.mask, p.mask) and not np.any(r.mask & l.mask)
    # near the hip each half owns its own leg column
    assert r.mask[30, 110:130].mean() > 0.8 and r.mask[30, 170:190].mean() < 0.2
    assert l.mask[30, 170:190].mean() > 0.8 and l.mask[30, 110:130].mean() < 0.2


def test_lr_extra_uses_rows_for_unbalanced_components():
    p = _legs_part(blob=True)
    out = L.stage_lr_extra([p], L.SplitReport())
    names = [q.name for q in out]
    assert names == ['leg_R', 'leg_L']
    r = out[0]
    assert 0.35 < r.area() / p.area() < 0.65


def test_split_leg_with_stray_blob():
    p = _legs_part(touch=False, blob=True)
    r, l = L.split_lr_rows(p, cx=150.0)
    ctx = L.SplitContext(canvas_hw=p.img.shape[:2])
    segs = L.split_leg(l, ctx, has_footwear=True)
    assert segs is not None and [s.name for s in segs] == ['thigh_L', 'lower_leg_L']
    assert np.array_equal(segs[0].mask | segs[1].mask, l.mask)
