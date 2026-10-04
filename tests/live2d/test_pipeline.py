import os.path as osp
import subprocess
import sys

import numpy as np
import pytest

import utils.live2d_split as L
from utils.live2d_psd import read_psd_layers

ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
UPSTREAM_BASE = 'a25a549'


def test_run_detailed_split_original_canvas(synth_dir, tmp_path):
    out = str(tmp_path / 'd.psd')
    r = L.run_detailed_split(synth_dir, original=osp.join(synth_dir, 'original.png'), out_psd=out)
    labels = [c[0] for c in r['counts']]
    assert labels[:2] == ['original semantic layers', 'after LR split']
    assert r['counts'][0][1] == 18 and r['counts'][1][1] == 24
    assert r['canvas_hw'] == [1024, 768]
    layers = read_psd_layers(out)
    assert len(layers) == len(r['names']) == r['counts'][-1][1]
    assert len(set(r['names'])) == len(r['names'])
    for l in layers:
        assert l['img'].shape == (1024, 768, 4)


def test_cli(synth_dir, tmp_path):
    out = str(tmp_path / 'cli.psd')
    p = subprocess.run([sys.executable, osp.join(ROOT, 'inference/scripts/live2d_detailed_split.py'),
                        '--srcd', synth_dir, '--out', out], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert 'after detailed split' in p.stdout and osp.exists(out)


def _git_show(rev, path):
    try:
        return subprocess.run(['git', '-C', ROOT, 'show', f'{rev}:{path}'], capture_output=True, text=True, check=True).stdout
    except Exception:
        pytest.skip('git history not available')


@pytest.mark.parametrize('path', ['common/utils/inference_utils.py', 'common/utils/io_utils.py', 'common/utils/cv.py',
                                  'common/utils/torchcv.py'])
def test_normal_mode_code_unchanged(path):
    """Checkbox/flag OFF must behave exactly as upstream: the normal pipeline modules are byte-identical."""
    assert open(osp.join(ROOT, path)).read() == _git_show(UPSTREAM_BASE, path)


def test_inference_psd_only_additions_behind_flag():
    old = _git_show(UPSTREAM_BASE, 'inference/scripts/inference_psd.py').splitlines()
    new = open(osp.join(ROOT, 'inference/scripts/inference_psd.py')).read().splitlines()
    it = iter(new)
    assert all(any(o == n for n in it) for o in old), 'an upstream line was modified or removed'
    added = [n for n in new if n not in old]
    code = [a for a in added if a.strip() and 'add_argument' not in a]
    assert code[0].strip() == 'if args.live2d_detailed_split:'
    assert all(c.startswith(' ' * 12) for c in code[1:])   # everything else is inside the if-block
