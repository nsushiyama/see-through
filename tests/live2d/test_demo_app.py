"""Run demo/app.py's real `inference` function with the GPU models stubbed out.

The stubbed apply_layerdiff/apply_marigold write a synthetic See-through output directory; the
stubbed further_extr writes a fixed normal PSD.  Verifies that the 'Live2D detailed split' checkbox
OFF returns exactly the normal outputs (and nothing else), and ON adds the detailed PSD/preview.
"""
import importlib.util
import os
import os.path as osp
import shutil
import sys
import types

import pytest

gr = pytest.importorskip('gradio')
ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))


class _Dummy:
    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    def __getattr__(self, k):
        return _Dummy()

    def __call__(self, *a, **k):
        return _Dummy()


def _install_stubs(calls):
    from synth import make_sample
    spaces = types.ModuleType('spaces')
    spaces.GPU = lambda duration=None: (lambda f: f)
    torch = types.ModuleType('torch')
    torch.bfloat16 = 'bf16'
    mods = {'spaces': spaces, 'torch': torch}
    for name, attrs in {
        'modules.layerdiffuse.diffusers_kdiffusion_sdxl': ['KDiffusionStableDiffusionXLPipeline'],
        'modules.layerdiffuse.layerdiff3d': ['UNetFrameConditionModel'],
        'modules.layerdiffuse.vae': ['TransparentVAE', 'TransparentVAEDecoder', 'TransparentVAEEncoder'],
        'modules.marigold': ['MarigoldDepthPipeline'],
    }.items():
        m = types.ModuleType(name)
        for a in attrs:
            setattr(m, a, _Dummy)
        mods[name] = m
    for pkg in ['modules', 'modules.layerdiffuse']:
        mods.setdefault(pkg, types.ModuleType(pkg))
    inf = types.ModuleType('utils.inference_utils')

    def apply_layerdiff(input_path, repo, save_dir, seed, resolution):
        calls.append('layerdiff')
        d = make_sample(save_dir, W=768, H=1024, resolution=1024, srcname='input')
        from PIL import Image
        Image.open(osp.join(d, 'original.png')).save(input_path)   # the "uploaded" image

    def apply_marigold(*a, **k):
        calls.append('marigold')

    def further_extr(saved, rotate, save_to_psd, tblr_split):
        calls.append(('further_extr', tblr_split))
        with open(saved + '.psd', 'wb') as f:
            f.write(b'NORMAL-PSD')
    inf.apply_layerdiff, inf.apply_marigold, inf.further_extr = apply_layerdiff, apply_marigold, further_extr
    mods['utils.inference_utils'] = inf
    tu = types.ModuleType('utils.torch_utils')
    tu.seed_everything = lambda s: None
    mods['utils.torch_utils'] = tu
    return mods


@pytest.fixture()
def app(monkeypatch):
    import utils  # real package (common/utils)  # noqa: F401
    calls = []
    for k, v in _install_stubs(calls).items():
        monkeypatch.setitem(sys.modules, k, v)
    spec = importlib.util.spec_from_file_location('demo_app_under_test', osp.join(ROOT, 'demo', 'app.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._move_to_gpu = lambda: None
    return mod, calls


def test_checkbox_off_is_normal_behaviour(app):
    mod, calls = app
    from PIL import Image
    img = Image.new('RGB', (768, 1024), 'white')
    out = mod.inference(img, 768, 42, False)          # positional call exactly like the original Space
    psd, gallery, live2d_psd, preview, info = out
    assert open(psd, 'rb').read() == b'NORMAL-PSD'
    assert live2d_psd is None and preview is None and info == ''
    assert ('further_extr', False) in calls
    assert all(t != 'src_img' for _, t in gallery)


def test_checkbox_on_adds_detailed_psd(app):
    mod, calls = app
    from PIL import Image
    from utils.live2d_psd import read_psd_layers
    img = Image.new('RGB', (768, 1024), 'white')
    psd, gallery, live2d_psd, preview, info = mod.inference(img, 768, 42, False, True, True)
    assert open(psd, 'rb').read() == b'NORMAL-PSD'     # normal output untouched
    layers = read_psd_layers(live2d_psd)
    assert len(layers) >= 40
    assert layers[0]['img'].shape == (1024, 768, 4)   # original canvas
    assert preview is not None and 'after LR split' in info


def test_ui_has_checkboxes(app):
    mod, _ = app
    labels = [getattr(b, 'label', None) for b in mod.demo.blocks.values()]
    assert 'Live2D detailed split' in labels and 'Show split preview' in labels
    assert 'Split left/right arms & legs' in labels
