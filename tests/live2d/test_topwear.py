import json
import os.path as osp

import numpy as np

import utils.live2d_split as L


def test_topwear_sleeves(synth_dir):
    fullpage, parts = L.load_semantic_layers(synth_dir)
    rep = L.SplitReport()
    ctx = L.SplitContext(canvas_hw=fullpage.shape[:2])
    parts = L.stage_lr_extra(L.stage_lr_split(parts, rep), rep)
    ctx.body_cx = L.body_center_x(parts)
    parts = L.stage_arms(parts, rep, ctx)
    out = L.stage_topwear(parts, rep, ctx)
    by = {p.name: p for p in out}
    for n in ['torso', 'upper_sleeve_L', 'lower_sleeve_L', 'upper_sleeve_R', 'lower_sleeve_R']:
        assert n in by, (n, rep.events)
    gt = json.load(open(osp.join(synth_dir, 'gt.json')))
    px, py = gt['pad_pos']
    at = lambda n, p: bool(by[n].mask[int(p[1]) + py, int(p[0]) + px])  # noqa: E731
    for s in 'LR':
        J = gt['joints_original'][f'arm_{s}']
        mid_upper = ((J['shoulder'][0] + J['elbow'][0]) / 2, (J['shoulder'][1] + J['elbow'][1]) / 2)
        mid_fore = ((J['elbow'][0] + J['wrist'][0]) / 2, (J['elbow'][1] + J['wrist'][1]) / 2)
        assert at(f'upper_sleeve_{s}', mid_upper) and not at('torso', mid_upper)
        assert at(f'lower_sleeve_{s}', mid_fore) and not at(f'upper_sleeve_{s}', mid_fore)
        assert at(f'upper_sleeve_{s}', J['elbow']) and at(f'lower_sleeve_{s}', J['elbow'])
    assert at('torso', (384, 440)) and not at('upper_sleeve_L', (384, 440))
    assert np.nonzero(by['upper_sleeve_R'].mask)[1].mean() < np.nonzero(by['upper_sleeve_L'].mask)[1].mean()
    assert all(by[n].group == 'Clothes' for n in ['torso', 'upper_sleeve_L'])
