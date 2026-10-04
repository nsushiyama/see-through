import json
import os.path as osp

import numpy as np
from PIL import Image


def test_synth_layout(synth_dir):
    from synth import V3_TAGS
    for t in V3_TAGS:
        assert osp.exists(osp.join(synth_dir, f'{t}.png'))
        assert osp.exists(osp.join(synth_dir, f'{t}_depth.png'))
    src = np.array(Image.open(osp.join(synth_dir, 'src_img.png')))
    assert src.shape == (1024, 1024, 4)
    for t in V3_TAGS:
        assert np.array(Image.open(osp.join(synth_dir, f'{t}.png'))).shape == src.shape
    info = json.load(open(osp.join(synth_dir, 'info.json')))
    assert set(info['parts']) == set(V3_TAGS)
    orig = Image.open(osp.join(synth_dir, 'original.png'))
    assert orig.size == (768, 1024)


def test_synth_loads_with_existing_load_parts(synth_dir):
    """The synthetic dir must be consumable by the unchanged repo loader."""
    from utils.io_utils import load_parts
    fullpage, infos, parts = load_parts(synth_dir)
    tags = {p['tag'] for p in parts}
    assert 'topwear' in tags and 'front hair' in tags
    assert 'tail' not in tags  # empty layers are dropped by load_part
    # Existing quirk (kept, not fixed): load_part() ignores layers whose pixels lie only in the
    # bottom/right 10% strip (`mask[:-p_test, :-p_test]`), so shoes at the very bottom are dropped.
    assert 'footwear' not in tags
    assert len(parts) == 17
