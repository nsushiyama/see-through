"""Live2D / Spine detailed split of an existing See-through output directory (CPU only).

    python inference/scripts/live2d_detailed_split.py --srcd workspace/layerdiff_output/<name> \
        [--original path/to/input.png] [--out out.psd] [--overlap 0.02] [--preview] [--no_restore] [--no_groups]

<srcd> is the directory written by apply_layerdiff + apply_marigold (contains src_img.png, <tag>.png,
<tag>_depth.png, info.json).  The normal PSD is not touched.
"""
import argparse
import json
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 'common'))

from utils.live2d_split import run_detailed_split  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--srcd', required=True)
    ap.add_argument('--original', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--overlap', type=float, default=0.02, help='joint overlap ratio of the long side (0.01-0.03)')
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--no_restore', action='store_true', help='do not restore original visible pixels')
    ap.add_argument('--no_groups', action='store_true')
    a = ap.parse_args()
    r = run_detailed_split(a.srcd, original=a.original, out_psd=a.out, overlap_ratio=a.overlap,
                           restore_visible=not a.no_restore, preview=a.preview, use_groups=not a.no_groups)
    print(json.dumps({k: r[k] for k in ('psd', 'counts', 'names', 'preview', 'canvas_hw')}, indent=1, ensure_ascii=False))
