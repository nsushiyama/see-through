"""
Split the layers of a See-through PSD into more, finer layers (e.g. 23 -> ~70).

Runs on CPU with numpy / OpenCV only, so a PSD from the online demo can be refined locally:

    python inference/scripts/fine_split_psd.py --srcp workspace/layerdiff_output/test_image.psd
    python inference/scripts/fine_split_psd.py --srcp in.psd --config fine70 --target_layers 90 --inpaint lama

The sidecar <name>_depth.psd is used for front/back ordering when present; without it the
PSD layer order is kept. Writes <name>_fine.psd and <name>_fine_depth.psd.
"""

import os.path as osp
import argparse

from utils.fine_split import apply_fine_split, load_psd_parts, save_parts_psd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcp', type=str, required=True, help='input PSD')
    parser.add_argument('--savep', type=str, default=None, help='output PSD, default <srcp>_fine.psd')
    parser.add_argument('--config', type=str, default='fine70', help='preset name under common/assets/fine_split or a json path')
    parser.add_argument('--target_layers', type=int, default=None, help='overrides target_layers of the config')
    parser.add_argument('--inpaint', type=str, default=None, choices=['cv2', 'lama', 'none'], help='overrides inpaint of the config')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    tag2pinfo, frame_size = load_psd_parts(args.srcp)
    apply_fine_split(tag2pinfo, args.config, target_layers=args.target_layers, inpaint=args.inpaint, seed=args.seed)
    savep = args.savep or osp.splitext(args.srcp)[0] + '_fine.psd'
    save_parts_psd(tag2pinfo, frame_size, savep)
    print(f'psd saved to {savep} ({len(tag2pinfo)} layers)')
