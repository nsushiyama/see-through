import os.path as osp

import numpy as np
from PIL import Image

import utils.live2d_split as L
from utils.live2d_psd import read_psd_layers


def _layers(synth_dir, tmp_path, restore):
    out = str(tmp_path / f'r{int(restore)}.psd')
    L.run_detailed_split(synth_dir, original=osp.join(synth_dir, 'original.png'), out_psd=out, restore_visible=restore)
    return read_psd_layers(out)


def _comp(layers):
    return L.composite([L.Part(l['name'], l['img'], 0, l['name']) for l in layers])


def test_recomposite_matches_original_better_after_restore(synth_dir, tmp_path):
    orig = np.array(Image.open(osp.join(synth_dir, 'original.png')).convert('RGBA'))
    a = _layers(synth_dir, tmp_path, False)
    b = _layers(synth_dir, tmp_path, True)
    ca, cb = _comp(a), _comp(b)
    m = (orig[..., 3] > 250) & (cb[..., 3] > 250)
    ea = np.abs(ca[..., :3].astype(int) - orig[..., :3].astype(int))[m].mean()
    eb = np.abs(cb[..., :3].astype(int) - orig[..., :3].astype(int))[m].mean()
    assert eb < 0.5 * ea, (ea, eb)
    # silhouette/alpha unchanged by the restore
    for la, lb in zip(a, b):
        assert la['name'] == lb['name']
        assert np.array_equal(la['img'][..., 3], lb['img'][..., 3])


def test_hidden_pixels_untouched(synth_dir, tmp_path):
    a = {l['name']: l['img'] for l in _layers(synth_dir, tmp_path, False)}
    b = {l['name']: l['img'] for l in _layers(synth_dir, tmp_path, True)}
    # upper arm under the sleeve and thighs under the skirt are hidden -> must stay the inpainted pixels
    for n, cover in [('upper_arm_L', 'upper_sleeve_L'), ('thigh_R', 'skirt_R')]:
        hidden = (a[n][..., 3] > 200) & (b[cover][..., 3] > 250)
        assert hidden.sum() > 100
        assert np.array_equal(a[n][hidden], b[n][hidden]), n
