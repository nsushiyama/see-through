import json
import os.path as osp

import numpy as np

import utils.live2d_split as L


def _cx(p):
    return np.nonzero(p.mask)[1].mean()


def test_existing_lr_rule_character_side(synth_dir):
    _, parts = L.load_semantic_layers(synth_dir)
    rep = L.SplitReport()
    out = L.stage_lr_split(parts, rep)
    names = {p.name for p in out}
    for b in ['arm', 'eye_white', 'iris', 'eyelash', 'eyebrow', 'ear']:
        assert f'{b}_L' in names and f'{b}_R' in names, b
    by = {p.name: p for p in out}
    for b in ['arm', 'eye_white', 'iris', 'eyebrow', 'ear']:
        # character RIGHT is on the SCREEN LEFT for a front-facing character
        assert _cx(by[f'{b}_R']) < _cx(by[f'{b}_L'])
        assert by[f'{b}_R'].side == 'character_right'
    # no pixel lost, no pixel duplicated
    src = {p.source: p for p in parts}
    for b, tag in [('arm', 'handwear'), ('iris', 'irides')]:
        r, l = by[f'{b}_R'].mask, by[f'{b}_L'].mask
        assert not np.any(r & l)
        assert np.array_equal(r | l, src[tag].mask)
        # RGBA values preserved exactly inside each half
        assert np.array_equal(by[f'{b}_R'].img[r], src[tag].img[r])
    # arm joints from gt: right arm elbow must be inside arm_R
    gt = json.load(open(osp.join(synth_dir, 'gt.json')))
    px, py = gt['pad_pos']
    ex, ey = gt['joints_original']['arm_R']['elbow']
    assert by['arm_R'].mask[ey + py, ex + px]
    assert len(out) == len(parts) + 6


def test_extra_lr_feet_legs(synth_dir):
    _, parts = L.load_semantic_layers(synth_dir)
    rep = L.SplitReport()
    out = L.stage_lr_extra(L.stage_lr_split(parts, rep), rep)
    by = {p.name: p for p in out}
    assert _cx(by['foot_R']) < _cx(by['foot_L'])
    assert _cx(by['leg_R']) < _cx(by['leg_L'])


def test_legs_connected_use_rows(synth_dir):
    _, parts = L.load_semantic_layers(synth_dir)
    leg = [p for p in parts if p.source == 'legwear'][0]
    m = leg.mask.copy()
    ys, xs = np.nonzero(m)
    y0 = ys.min()
    leg.img[y0:y0 + 40, xs.min():xs.max()] = (20, 20, 30, 255)  # tights waistband joins the legs
    rep = L.SplitReport()
    out = L.stage_lr_extra(parts, rep)
    by = {p.name: p for p in out}
    # row-wise gap cut (falls back to body mid-line when no gap row is found)
    assert by['leg_R'].method in ('lr_rows', 'lr_midline')
    assert _cx(by['leg_R']) < _cx(by['leg_L'])


def test_suspicious_legwear_kept(synth_dir):
    _, parts = L.load_semantic_layers(synth_dir)
    leg = [p for p in parts if p.source == 'legwear'][0]
    leg.img[..., 3] = 255  # background leakage like the real HF sample
    rep = L.SplitReport()
    out = L.stage_lr_extra(parts, rep)
    assert any(p.name == 'leg' for p in out)
    assert any('background leakage' in e for e in rep.events)
