import json
import os.path as osp

import numpy as np

import utils.live2d_split as L


def _run(d):
    fullpage, parts = L.load_semantic_layers(d)
    rep = L.SplitReport()
    ctx = L.SplitContext(canvas_hw=fullpage.shape[:2])
    parts = L.stage_lr_split(parts, rep)
    return L.stage_hair(parts, rep, ctx), rep, parts


def test_front_hair_strands(synth_dir):
    out, rep, _ = _run(synth_dir)
    names = [p.name for p in out if p.source == 'front hair']
    for n in ['front_hair_center', 'front_hair_R_01', 'front_hair_R_02', 'front_hair_L_01', 'front_hair_L_02',
              'side_hair_R', 'side_hair_L']:
        assert n in names, (n, names, rep.events)
    by = {p.name: p for p in out}
    xm = lambda n: np.nonzero(by[n].mask)[1].mean()  # noqa: E731
    assert xm('side_hair_R') < xm('front_hair_R_02') < xm('front_hair_R_01') < xm('front_hair_center') \
        < xm('front_hair_L_01') < xm('front_hair_L_02') < xm('side_hair_L')
    # the front hair is ONE connected region: the split must not rely on connected components
    src = [p for p in L.load_semantic_layers(synth_dir)[1] if p.source == 'front hair'][0]
    assert len(L.components(src.mask)) == 1
    # every strand tip is (almost) entirely owned by one part -> cuts go between strands, not through them
    gt = json.load(open(osp.join(synth_dir, 'gt.json')))
    px, py = gt['pad_pos']
    bangs = [by[n] for n in names if n.startswith('front_hair')]
    for tx, ty in gt['joints_original']['front_hair_tips']:
        tip = np.zeros(src.mask.shape, bool)
        tip[ty + py - 6: ty + py + 2, tx + px - 6: tx + px + 6] = True
        tip &= src.mask
        share = max((b.mask & tip).sum() for b in bangs) / max(tip.sum(), 1)
        assert share > 0.9, (tx, ty, share)


def test_back_hair_sectors(synth_dir):
    out, rep, _ = _run(synth_dir)
    names = [p.name for p in out if p.source == 'back hair']
    assert 'back_hair_center' in names
    assert sum(n.startswith('back_hair_R_') for n in names) >= 2
    assert sum(n.startswith('back_hair_L_') for n in names) >= 2
    by = {p.name: p for p in out}
    assert np.nonzero(by['back_hair_R_01'].mask)[1].mean() < np.nonzero(by['back_hair_L_01'].mask)[1].mean()


def test_hair_fallback_without_face(synth_dir):
    fullpage, parts = L.load_semantic_layers(synth_dir)
    parts = [p for p in parts if p.source != 'face']
    out = L.stage_hair(parts, L.SplitReport(), L.SplitContext(canvas_hw=fullpage.shape[:2]))
    assert {'front_hair', 'back_hair'} <= {p.name for p in out}
