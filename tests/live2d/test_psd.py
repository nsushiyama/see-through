import numpy as np

import utils.live2d_split as L
from utils.live2d_psd import save_live2d_psd, read_psd_layers


def _parts(synth_dir):
    _, parts = L.load_semantic_layers(synth_dir)
    rep = L.SplitReport()
    return L.stage_lr_extra(L.stage_lr_split(parts, rep), rep)


def test_psd_roundtrip_groups_nonsquare(synth_dir, tmp_path):
    parts = _parts(synth_dir)
    # export to the NON-SQUARE original canvas (768 x 1024)
    H, W = 1024, 768
    exp = [L.Part(p.name, L.to_original_canvas(p.img, (H, W)), p.depth, p.source, p.group, p.side, p.method) for p in parts]
    psdp = str(tmp_path / 'detailed.psd')
    meta = save_live2d_psd(psdp, exp, (H, W))
    assert meta['mode'] == 'groups'
    back = read_psd_layers(psdp)
    assert [b['name'] for b in back] == [p.name for p in exp]           # z-order preserved
    from psd_tools import PSDImage
    assert PSDImage.open(psdp).size == (W, H)
    for b, p in zip(back, exp):
        assert b['img'].shape == (H, W, 4)
        a = p.img[..., 3]
        assert np.array_equal(b['img'][..., 3], a), b['name']           # alpha exact, no offset
        vis = a > 0
        assert np.array_equal(b['img'][..., :3][vis], p.img[..., :3][vis]), b['name']
        assert b['group'] == p.group or b['group'].startswith(p.group + '_')
    groups = {b['group'] for b in back}
    assert {'Hair', 'Face', 'Body', 'Clothes'} <= {g.split('_')[0] for g in groups}


def test_psd_prefix_mode(synth_dir, tmp_path):
    parts = _parts(synth_dir)[:5]
    psdp = str(tmp_path / 'flat.psd')
    meta = save_live2d_psd(psdp, parts, parts[0].img.shape[:2], use_groups=False)
    back = read_psd_layers(psdp)
    assert meta['mode'] == 'prefix'
    assert [b['name'] for b in back] == [f'{p.group}_{p.name}' for p in parts]
    assert all(b['group'] is None for b in back)
