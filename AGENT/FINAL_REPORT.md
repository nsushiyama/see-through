# FINAL REPORT — Live2D detailed split

## Status
**COMPLETE** on branch `live2d-detailed-split` of fork `nsushiyama/see-through`
(upstream `shitagaki-lab/see-through` untouched).

## What was built
Post-processing after See-through semantic RGBA (+ existing LR split):
hair sectors, eyes/ears/feet CC LR, arms/legs with joint overlap, topwear sleeves when
protruding, skirt/bottomwear sectors, accessories, visible-pixel restore, bg-leak cleanup,
PSD groups, Gradio checkboxes.

## Changed / added files (high level)
- `common/utils/live2d_split.py` — core pipeline
- `common/utils/live2d_psd.py` — grouped full-canvas PSD
- `inference/scripts/live2d_detailed_split.py`, `inference_psd.py` flag
- `demo/app.py` — Live2D detailed split + Show split preview
- `tests/live2d/**` — synth, unit, demo stub, REAL_IMAGE.md, artifacts
- `AGENT/**` — goal/state/todo/worklog/analysis

## How to run
```bash
# unit
PYTHONPATH=common:. python -m pytest tests/live2d -q

# CLI on a See-through output dir (src_img.png + <tag>.png + depths)
python inference/scripts/live2d_detailed_split.py /path/to/srcd --out out.psd --preview

# demo (needs GPU + See-through weights for full inference; checkbox OFF = upstream)
cd demo && python app.py
```

## HF Space deps to add for the checkbox path
`psd-tools`, `opencv-python-headless`, `scipy`, `scikit-image` (post-process only; no new SAM dep).

## Real-image results (Space 24yearsold/see-through-demo @768)
| image | stages | layers |
|-------|--------|--------|
| img1 skirt (primary) | 17→23→45 | OK LR/overlap/offset/inpaint |
| img0 T-shirt | 17→23→44 | OK |
| img2 | 17→23→45 | sleeves+headwear |
| img3 | 15→19→37 | fewer parts |

## Known limitations
- Layer count often 37–45 (<50–80 target) without inventing parts — by design.
- Short / non-protruding sleeves → topwear kept.
- Same-colour puffy sleeves may remain on upper_arm.
- No GPU on agent box: inference via HF Space; post-process local.
- Optional TASK-022 (SemanticSam/pose) not done.
