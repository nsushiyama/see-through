import numpy as np

import utils.live2d_split as L


def _ctx_parts(d):
    fullpage, parts = L.load_semantic_layers(d)
    rep = L.SplitReport()
    parts = L.stage_lr_split(parts, rep)
    ctx = L.SplitContext(canvas_hw=fullpage.shape[:2], body_cx=L.body_center_x(parts))
    return parts, rep, ctx


def xm(p):
    return np.nonzero(p.mask)[1].mean()


def test_skirt_sectors(synth_dir):
    parts, rep, ctx = _ctx_parts(synth_dir)
    by = {p.name: p for p in L.stage_bottomwear(parts, rep, ctx)}
    assert xm(by['skirt_R']) < xm(by['skirt_center']) < xm(by['skirt_L'])


def test_pants_split_midline(synth_dir):
    parts, rep, ctx = _ctx_parts(synth_dir)
    bw = [p for p in parts if p.source == 'bottomwear'][0]
    ys, xs = np.nonzero(bw.mask)
    c = int(ctx.body_cx)
    bw.img[int(ys.max() - 0.4 * np.ptp(ys)):, c - 8:c + 8, 3] = 0   # cut a leg gap -> shorts
    by = {p.name: p for p in L.stage_bottomwear(parts, rep, ctx)}
    assert 'bottomwear_R' in by and xm(by['bottomwear_R']) < xm(by['bottomwear_L'])


def test_accessories(synth_dir):
    parts, rep, ctx = _ctx_parts(synth_dir)
    by = {p.name: p for p in L.stage_accessories(parts, rep, ctx)}
    # hair ornaments: two separate pieces (screen-left one = character right)
    assert 'headwear_R' in by and 'headwear_L' in by
    assert xm(by['headwear_R']) < xm(by['headwear_L'])
    # neck ribbon is ONE connected shape -> main + tails by shape
    assert {'neckwear_main', 'neckwear_tail_R', 'neckwear_tail_L'} <= set(by), rep.events
    assert xm(by['neckwear_tail_R']) < xm(by['neckwear_tail_L'])
    assert np.nonzero(by['neckwear_tail_L'].mask)[0].mean() > np.nonzero(by['neckwear_main'].mask)[0].mean()


def test_face_parts_present(synth_dir):
    parts, rep, ctx = _ctx_parts(synth_dir)
    names = {p.name for p in parts}
    assert {'face', 'neck', 'ear_L', 'ear_R', 'nose', 'mouth'} <= names
