import os.path as osp
import sys

import pytest

ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
for p in (osp.join(ROOT, 'common'), osp.join(ROOT, 'tests', 'live2d')):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope='session')
def synth_dir(tmp_path_factory):
    from synth import make_sample
    return make_sample(str(tmp_path_factory.mktemp('synth')), W=768, H=1024, resolution=1024)


@pytest.fixture(scope='session')
def synth_dir_lowres(tmp_path_factory):
    """Inference resolution smaller than the original (exercise rescale on export)."""
    from synth import make_sample
    return make_sample(str(tmp_path_factory.mktemp('synth_lr')), W=768, H=1024, resolution=768)
