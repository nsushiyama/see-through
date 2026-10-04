import json
import os.path as osp

import numpy as np
from PIL import Image

import utils.live2d_split as L


def test_loader_full_canvas_keeps_bottom_layers(synth_dir):
    fullpage, parts = L.load_semantic_layers(synth_dir)
    assert fullpage.shape == (1024, 1024, 4)
    tags = {p.source for p in parts}
    assert 'footwear' in tags            # unlike io_utils.load_part, nothing is dropped
    assert 'tail' not in tags            # empty layers skipped
    assert len(parts) == 18
    for p in parts:
        assert p.img.shape == fullpage.shape and p.img.dtype == np.uint8
        raw = np.array(Image.open(osp.join(synth_dir, p.source + '.png')))
        assert np.array_equal(raw, p.img)  # untouched, not cropped
    d = [p.depth for p in parts]
    assert d == sorted(d, reverse=True)   # far -> near


def test_overlap_px_scales_with_resolution():
    assert L.overlap_px((1024, 768)) == round(1024 * 0.02)
    assert L.overlap_px((2048, 1536)) == round(2048 * 0.02)
    assert L.overlap_px((1000, 1000), ratio=0.5) == 30   # clamped to 3%
    assert L.overlap_px((1000, 1000), ratio=0.001) == 10  # clamped to 1%


def test_canvas_roundtrip_matches_original_geometry(synth_dir):
    gt = json.load(open(osp.join(synth_dir, 'gt.json')))
    H, W = gt['original_size']
    orig = np.array(Image.open(osp.join(synth_dir, 'original.png')).convert('RGBA'))
    sq = L.from_original_canvas(orig, 1024)
    src = np.array(Image.open(osp.join(synth_dir, 'src_img.png')).convert('RGBA'))
    assert sq.shape == src.shape
    back = L.to_original_canvas(src, (H, W))
    assert back.shape == (H, W, 4)
    # same resolution here (sz == R) -> pure un-padding, must be exact
    assert np.array_equal(back, orig)


def test_canvas_roundtrip_rescaled(synth_dir_lowres):
    gt = json.load(open(osp.join(synth_dir_lowres, 'gt.json')))
    H, W = gt['original_size']
    face = np.array(Image.open(osp.join(synth_dir_lowres, 'face.png')))
    assert face.shape[:2] == (768, 768)
    back = L.to_original_canvas(face, (H, W))
    assert back.shape == (H, W, 4)
    # centroid of the face must land where it is in the original (within ~1px)
    m = back[..., 3] > 127
    ys, xs = np.nonzero(m)
    assert abs(xs.mean() - W / 2) < 1.5
    assert abs(ys.mean() - 210) < 2.0


def test_composite_of_layers_reproduces_src(synth_dir):
    fullpage, parts = L.load_semantic_layers(synth_dir)
    comp = L.composite(parts)
    a = comp[..., 3] > 200
    diff = np.abs(comp[..., :3].astype(int) - fullpage[..., :3].astype(int))[a]
    assert np.percentile(diff, 99) <= 12   # src has +-6 detail noise


def _part(mask, name='x'):
    img = np.zeros(mask.shape + (4,), np.uint8)
    img[mask] = (200, 100, 50, 255)
    return L.Part(name, img, 0.5, 'topwear')


def test_safe_split_fallback_keeps_original():
    m = np.zeros((64, 64), bool); m[10:50, 10:50] = True
    src = _part(m, 'topwear')
    rep = L.SplitReport()

    def boom(p):
        raise RuntimeError('mask provider failed')
    assert L.safe_split(src, boom, rep, 't')[0] is src

    def lossy(p):  # loses half the pixels
        k = np.zeros_like(m); k[10:30, 10:50] = True
        return [p.derive('a', k), p.derive('b', k & False | (np.arange(64)[:, None] == 11) & m)]
    assert L.safe_split(src, lossy, rep, 't')[0] is src

    def good(p):
        k = np.zeros_like(m); k[:30] = True
        return [p.derive('a', k & m), p.derive('b', ~k & m)]
    out = L.safe_split(src, good, rep, 't')
    assert [o.name for o in out] == ['a', 'b']
    assert sum('fallback' in e for e in rep.events) == 2
