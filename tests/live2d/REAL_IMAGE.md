# Real-image verification (TASK-020)

Source: Hugging Face Space `24yearsold/see-through-demo` (ZeroGPU, anonymous).
Fetch: `python tests/live2d/fetch_hf_space.py IMG OUTDIR` then `python tests/live2d/psd_to_srcdir.py OUTDIR`
(PSD layers → full-canvas `<tag>.png`; gallery webp is lossy so PSD is preferred).

Test images = Space `common/assets/test_image{,1,2,3}.png` (928×1232), resolution=768, seed=42.

## Stage counts

| image | semantic | after LR | after detailed | notes |
|-------|----------|----------|----------------|-------|
| img0 (T-shirt / shorts) | 17 | 23 | 44 | topwear kept (short sleeves not protruding) |
| img1 (skirt / off-shoulder) **primary** | 17 | 23 | 45 | legwear had 73% canvas grey sheet → bg cleanup 7%; skirt_R/C/L |
| img2 | 17 | 23 | 45 | sleeves + headwear tails |
| img3 | 15 | 19 | 37 | arm LR skipped (single blob); fewer face parts |

## Quality checklist (img0 + img1 automated)

1. PSD size == input (928×1232) — OK
2. Every raster layer full-canvas offset (0,0) — OK (0 bad)
3. Alpha preserved (RGBA layers) — OK
4. Hidden/inpainted pixels kept (arm ∩ opaque topwear > 0) — OK (img1: 7087 px)
5. Recomposition uses visible restore from original — OK (img1: 220975 px feather 5)
6. Character-left / character-right correct (iris/arm/thigh screen-x: R < L) — OK
7. Joint overlap ≥ ov/2 (ov = 2% long side ≈ 15 px @768) — OK for elbow/wrist/knee/ankle
8. Fallback: topwear kept when sleeve split declines; earwear kept when CC finds nothing — OK
9. OFF mode: `normal.psd` from Space unchanged; detailed path only when checkbox on (unit: test_demo_app / test_pipeline) — OK
10. Layer count 37–45 (below 50–80 target): **semantic correctness preferred over padding**; no fake layers. Short/absent sleeves and empty accessory tags are the main gap.

## Artifacts

`tests/live2d/artifacts/img*_r768_{report.json,preview.jpg}` — stage logs, names, compact mask preview.
Full PSDs live under `/mnt/stx/work/real/` on the agent box (too large for git).

## Known model-side quirks handled

- **legwear background sheet**: `stage_bg_cleanup` strips input-background pixels from suspicious layers before LR/limb split.
- **touching knees**: `split_lr_rows` (gap per scanline) instead of a hard mid-line.
- **handwear = bare arms + sleeves**: arm split + optional cloth/skin sleeve peel on the arm layer.
