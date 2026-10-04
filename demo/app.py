# Gradio web demo for See-through.
# Derived from the official Hugging Face Space `24yearsold/see-through-demo` (app.py @ 9cc6ee4b,
# Space license: MIT, https://huggingface.co/spaces/24yearsold/see-through-demo), which wraps
# https://github.com/shitagaki-lab/see-through (Apache-2.0). Unmodified copy: demo/app_space_original.py
# Local changes are marked with "# [fork]".
import spaces
import gradio as gr
import os
import sys
import time
import tempfile
import shutil
import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # [fork] app lives in demo/, repo root is parent
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "common"))

from PIL import Image

REPO_LAYERDIFF = "layerdifforg/seethroughv0.0.2_layerdiff3d"
REPO_DEPTH = "24yearsold/seethroughv0.0.1_marigold"


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------- Preload models to CPU at startup ---------------
_log("Preloading LayerDiff pipeline to CPU...")
from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
from modules.layerdiffuse.vae import TransparentVAE, TransparentVAEDecoder, TransparentVAEEncoder

_trans_vae = TransparentVAE.from_pretrained(REPO_LAYERDIFF, subfolder="trans_vae")
_unet_ld = UNetFrameConditionModel.from_pretrained(REPO_LAYERDIFF, subfolder="unet")
_layerdiff_pipe = KDiffusionStableDiffusionXLPipeline.from_pretrained(
    REPO_LAYERDIFF, trans_vae=_trans_vae, unet=_unet_ld, scheduler=None
)
_log("LayerDiff pipeline loaded to CPU.")

_log("Preloading Marigold pipeline to CPU...")
from modules.marigold import MarigoldDepthPipeline

_unet_mg = UNetFrameConditionModel.from_pretrained(REPO_DEPTH, subfolder="unet")
_marigold_pipe = MarigoldDepthPipeline.from_pretrained(REPO_DEPTH, unet=_unet_mg)
_log("Marigold pipeline loaded to CPU.")

_models_on_gpu = False

from utils.inference_utils import apply_layerdiff, apply_marigold, further_extr
from utils.torch_utils import seed_everything
import utils.inference_utils as _inf


def _move_to_gpu():
    global _models_on_gpu
    if _models_on_gpu:
        _log("Models already on GPU, skipping transfer.")
        return

    t0 = time.time()
    _log("Moving LayerDiff to CUDA bf16...")
    _layerdiff_pipe.vae.to(dtype=torch.bfloat16, device="cuda")
    _layerdiff_pipe.trans_vae.to(dtype=torch.bfloat16, device="cuda")
    _layerdiff_pipe.unet.to(dtype=torch.bfloat16, device="cuda")
    _layerdiff_pipe.text_encoder.to(dtype=torch.bfloat16, device="cuda")
    _layerdiff_pipe.text_encoder_2.to(dtype=torch.bfloat16, device="cuda")
    _log(f"LayerDiff on GPU ({time.time() - t0:.1f}s)")

    t0 = time.time()
    _log("Moving Marigold to CUDA bf16...")
    _marigold_pipe.to(device="cuda", dtype=torch.bfloat16)
    _log(f"Marigold on GPU ({time.time() - t0:.1f}s)")

    # Inject into inference_utils globals so apply_* functions skip their own loading
    _inf.layerdiff_pipeline = _layerdiff_pipe
    _inf.marigold_pipeline = _marigold_pipe

    _models_on_gpu = True


_SKIP_TAGS = {"src_img", "src_head", "reconstruction"}

def _collect_layer_gallery(saved_dir):
    """Collect layer PNGs as (image, label) tuples for the gallery."""
    gallery = []
    for f in sorted(os.listdir(saved_dir)):
        if not f.endswith(".png"):
            continue
        tag = f[:-4]
        if tag.endswith("_depth") or tag in _SKIP_TAGS:
            continue
        img = Image.open(os.path.join(saved_dir, f))
        gallery.append((img, tag))
    return gallery


@spaces.GPU(duration=120)
def inference(image: Image.Image, resolution: int = 768, seed: int = 42, tblr_split: bool = False):
    t_start = time.time()
    if image is None:
        raise gr.Error("Please upload an image.")

    # Snap to nearest multiple of 64 for clean latent dimensions
    resolution = max(64, min(resolution, 1280))
    resolution = round(resolution / 64) * 64
    _log(f"Resolution: {resolution}, Seed: {seed}, Image: {image.size}")

    _move_to_gpu()
    seed_everything(seed)

    tmpdir = tempfile.mkdtemp(prefix="seethrough_")
    try:
        input_path = os.path.join(tmpdir, "input.png")
        image.save(input_path)

        t0 = time.time()
        _log("Running LayerDiff...")
        apply_layerdiff(
            input_path, REPO_LAYERDIFF,
            save_dir=tmpdir, seed=seed, resolution=resolution,
        )
        _log(f"LayerDiff done ({time.time() - t0:.1f}s)")

        t0 = time.time()
        _log("Running Marigold depth...")
        apply_marigold(
            input_path, REPO_DEPTH,
            save_dir=tmpdir, seed=seed, resolution=resolution,
        )
        _log(f"Marigold done ({time.time() - t0:.1f}s)")

        saved = os.path.join(tmpdir, "input")

        # Collect gallery before PSD assembly (further_extr may modify files)
        gallery = _collect_layer_gallery(saved)

        t0 = time.time()
        _log("Running PSD assembly...")
        further_extr(saved, rotate=False, save_to_psd=True, tblr_split=tblr_split)
        _log(f"PSD assembly done ({time.time() - t0:.1f}s)")

        psd_path = saved + ".psd"
        if os.path.exists(psd_path):
            output_path = os.path.join(
                tempfile.gettempdir(), "seethrough_output.psd"
            )
            shutil.copy2(psd_path, output_path)
            _log(f"Total inference time: {time.time() - t_start:.1f}s")
            return output_path, gallery

        raise gr.Error("PSD generation failed — no output file produced.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


with gr.Blocks(title="See-through: Layer Decomposition") as demo:
    gr.Markdown(
        "# See-through: Single-image Layer Decomposition for Anime Characters\n\n"
        '<a href="https://github.com/shitagaki-lab/see-through" target="_blank">GitHub</a> | '
        '<a href="https://arxiv.org/abs/2602.03749" target="_blank">Paper (arXiv:2602.03749)</a>\n\n'
        "Upload an anime character illustration to decompose it into "
        "fully-inpainted semantic layers with depth ordering, "
        "exported as a layered PSD file.\n\n"
        "**Note:** 768 resolution is recommended for ZeroGPU free tier. "
        "Higher resolutions may timeout or exhaust your daily quota. "
        'For best quality, clone the <a href="https://github.com/shitagaki-lab/see-through" target="_blank">full repo</a> '
        "and run `inference_psd.py` locally. "
        'We also have a <a href="https://github.com/jtydhr88/ComfyUI-See-through" target="_blank">ComfyUI Node Extension</a> '
        "and other community extensions — check our repo for more details.\n\n"
        "**Disclaimer:** This demo uses a newer model checkpoint and "
        "may not fully reproduce identical results reported in the paper."
    )
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload image (non-square images will be padded)")
            resolution = gr.Slider(
                minimum=768, maximum=1280, value=768, step=64,
                label="Resolution",
                info="768 recommended for ZeroGPU free tier. Higher resolutions may timeout or use up your daily quota quickly.",
            )
            seed = gr.Slider(minimum=0, maximum=9999, value=42, step=1, label="Seed")
            tblr_split = gr.Checkbox(
                value=False,
                label="Split left/right arms & legs",
                info="Separate left and right limbs into individual layers. Useful if the default output glues them together.",
            )
            run_btn = gr.Button("Run", variant="primary")
        with gr.Column(scale=2):
            psd_output = gr.File(label="Download layered PSD")
            gallery_output = gr.Gallery(label="Separated layers", columns=4, height="auto")

    run_btn.click(
        fn=inference,
        inputs=[input_image, resolution, seed, tblr_split],
        outputs=[psd_output, gallery_output],
    )
    _example = os.path.join(_root, "common/assets/test_image.png")
    if os.path.exists(_example):  # [fork] example image only ships with the HF Space
        gr.Examples(
            examples=[[_example, 768, 42, False]],
            inputs=[input_image, resolution, seed, tblr_split],
        )

if __name__ == "__main__":
    demo.launch()
