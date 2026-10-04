import json
import os.path as osp

import numpy as np

import utils.live2d_split as L


def _run(synth_dir, drop_footwear=False):
    fullpage, parts = L.load_semantic_layers(synth_dir)
    if drop_footwear:
        parts = [p for p in parts if p.source != 'footwear']
    rep = L.SplitReport()
    ctx = L.SplitContext(canvas_hw=fullpage.shape[:2])
    parts = L.stage_lr_extra(L.stage_lr_split(parts, rep), rep)
    return {p.name: p for p in L.stage_legs(parts, rep, ctx)}, ctx, rep


def test_leg_split_knee_overlap(synth_dir):
    by, ctx, rep = _run(synth_dir)
    gt = json.load(open(osp.join(synth_dir, 'gt.json')))
    px, py = gt['pad_pos']
    for s in 'LR':
        assert f'thigh_{s}' in by and f'lower_leg_{s}' in by, rep.events
        assert f'foot_{s}' in by and by[f'foot_{s}'].source == 'footwear'
        J = {k: (v[0] + px, v[1] + py) for k, v in gt['joints_original'][f'leg_{s}'].items()}
        at = lambda n, p: bool(by[n].mask[int(p[1]), int(p[0])])  # noqa: E731
        assert at(f'thigh_{s}', J['knee']) and at(f'lower_leg_{s}', J['knee'])
        assert at(f'lower_leg_{s}', J['ankle']) and not at(f'thigh_{s}', J['ankle'])
        mid = ((J['hip'][0] + J['knee'][0]) / 2, (J['hip'][1] + J['knee'][1]) / 2 - 30)
        assert at(f'thigh_{s}', mid) and not at(f'lower_leg_{s}', mid)
    assert np.nonzero(by['thigh_R'].mask)[1].mean() < np.nonzero(by['thigh_L'].mask)[1].mean()


def test_leg_without_footwear_gets_foot(synth_dir):
    by, _, _ = _run(synth_dir, drop_footwear=True)
    for s in 'LR':
        assert by[f'foot_{s}'].source == 'legwear'
