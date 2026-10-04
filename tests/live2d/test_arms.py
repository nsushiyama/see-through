import json
import os.path as osp

import numpy as np

import utils.live2d_split as L


def _run(synth_dir):
    fullpage, parts = L.load_semantic_layers(synth_dir)
    rep = L.SplitReport()
    parts = L.stage_lr_extra(L.stage_lr_split(parts, rep), rep)
    ctx = L.SplitContext(canvas_hw=fullpage.shape[:2])
    return L.stage_arms(parts, rep, ctx), rep, ctx


def test_arm_split_joints_and_overlap(synth_dir):
    out, rep, ctx = _run(synth_dir)
    by = {p.name: p for p in out}
    gt = json.load(open(osp.join(synth_dir, 'gt.json')))
    px, py = gt['pad_pos']
    for s in ('L', 'R'):
        for n in ('upper_arm', 'forearm', 'hand'):
            assert f'{n}_{s}' in by, (n, s, rep.events)
            assert by[f'{n}_{s}'].img.shape == (1024, 1024, 4)
        J = {k: (v[0] + px, v[1] + py) for k, v in gt['joints_original'][f'arm_{s}'].items()}
        ua, fa, hd = by[f'upper_arm_{s}'].mask, by[f'forearm_{s}'].mask, by[f'hand_{s}'].mask
        at = lambda m, p: bool(m[int(p[1]), int(p[0])])  # noqa: E731
        assert at(ua, J['shoulder']) and not at(fa, J['shoulder'])
        assert at(ua, J['elbow']) and at(fa, J['elbow']), 'elbow must be in the overlap'
        assert at(fa, J['wrist']) and at(hd, J['wrist']), 'wrist must be in the overlap'
        assert at(hd, J['hand']) and not at(ua, J['hand'])
        # overlap band length along the arm ~ 2 * ov
        inter = ua & fa
        ys, xs = np.nonzero(inter)
        span = np.hypot(np.ptp(xs), np.ptp(ys))
        assert 1.2 * ctx.ov <= span <= 4.5 * ctx.ov
        # no sleeve cloth in a bare synthetic arm
        assert f'upper_sleeve_{s}' not in by
    # L/R not swapped: right arm parts on the screen left
    assert np.nonzero(by['forearm_R'].mask)[1].mean() < np.nonzero(by['forearm_L'].mask)[1].mean()


def test_arm_pixels_preserved(synth_dir):
    out, _, _ = _run(synth_dir)
    _, parts = L.load_semantic_layers(synth_dir)
    hw = [p for p in parts if p.source == 'handwear'][0]
    union = np.zeros(hw.mask.shape, bool)
    for p in out:
        if p.source == 'handwear':
            union |= p.mask
            m = p.mask
            assert np.array_equal(p.img[m], hw.img[m])  # RGBA untouched inside the part
    assert np.array_equal(union, hw.mask)


def test_arm_overlap_scales_with_resolution(synth_dir_lowres, synth_dir):
    _, _, c1 = _run(synth_dir)
    _, _, c2 = _run(synth_dir_lowres)
    assert c1.ov == round(1024 * 0.02) and c2.ov == round(768 * 0.02)


def test_sleeve_inside_handwear_detected_by_colour(synth_dir):
    fullpage, parts = L.load_semantic_layers(synth_dir)
    hw = [p for p in parts if p.source == 'handwear'][0]
    gt = json.load(open(osp.join(synth_dir, 'gt.json')))
    px, py = gt['pad_pos']
    ey = gt['joints_original']['arm_L']['elbow'][1] + py
    xs = np.arange(1024)[None, :]
    ys = np.arange(1024)[:, None]
    cx = 384 + px
    puff = hw.mask & (ys < ey - 20) & (xs > cx)   # character-LEFT upper arm covered by a blue sleeve
    hw.img[puff, :3] = (60, 90, 200)
    rep = L.SplitReport()
    parts = L.stage_lr_extra(L.stage_lr_split(parts, rep), rep)
    out = L.stage_arms(parts, rep, L.SplitContext(canvas_hw=fullpage.shape[:2]))
    by = {p.name: p for p in out}
    assert 'upper_sleeve_L' in by and by['upper_sleeve_L'].group == 'Clothes'
    sl = by['upper_sleeve_L'].mask
    assert (sl & puff).sum() / puff.sum() > 0.9
    assert (sl & ~puff).sum() < 0.05 * sl.sum()
    assert 'upper_sleeve_R' not in by      # bare right arm: no false sleeve
