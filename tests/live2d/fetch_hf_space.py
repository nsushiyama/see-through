"""Fetch REAL See-through outputs from the official Hugging Face Space (no local GPU needed).

    TMPDIR=/mnt/stx/tmp python tests/live2d/fetch_hf_space.py IMAGE OUT_DIR [--resolution 768] [--seed 42] [--tblr_split]

Calls ``24yearsold/see-through-demo`` /inference via gradio_client and stores:
  OUT_DIR/normal.psd           normal-mode PSD exactly as the Space returns it
  OUT_DIR/gallery/<tag>.<ext>  per-tag full-canvas layers from the Space gallery
  OUT_DIR/input.png            the input image
  OUT_DIR/meta.json            call parameters and timings
Then ``tests/live2d/psd_to_srcdir.py`` converts normal.psd into a See-through output directory
(full-canvas <tag>.png + depth) usable by the detailed split.
"""
import argparse
import json
import os
import os.path as osp
import shutil
import time

SPACE = '24yearsold/see-through-demo'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('out_dir')
    ap.add_argument('--resolution', type=int, default=768)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tblr_split', action='store_true')
    a = ap.parse_args()
    from gradio_client import Client, handle_file
    os.makedirs(osp.join(a.out_dir, 'gallery'), exist_ok=True)
    shutil.copy(a.image, osp.join(a.out_dir, 'input.png'))
    t0 = time.time()
    client = Client(SPACE)
    psd, gallery = client.predict(handle_file(a.image), a.resolution, a.seed, a.tblr_split, api_name='/inference')
    shutil.copy(psd, osp.join(a.out_dir, 'normal.psd'))
    names = []
    for item in gallery:
        src = item['image'] if isinstance(item['image'], str) else item['image']['path']
        cap = item.get('caption') or osp.splitext(osp.basename(src))[0]
        dst = osp.join(a.out_dir, 'gallery', cap + osp.splitext(src)[1])
        shutil.copy(src, dst)
        names.append(osp.basename(dst))
    meta = dict(space=SPACE, resolution=a.resolution, seed=a.seed, tblr_split=a.tblr_split,
                seconds=round(time.time() - t0, 1), gallery=names)
    json.dump(meta, open(osp.join(a.out_dir, 'meta.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
