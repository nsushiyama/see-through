# STATE

STATUS: COMPLETE

CURRENT_PHASE:
done

LAST_COMPLETED_TASK:
TASK-021 final checks

CURRENT_TASK:
(none)

NEXT_TASK:
(optional) TASK-022 SemanticSam / pose mask provider hook on GPU

BLOCKERS:
- none

LAST_GOOD_COMMIT:
(see `git log -1` on live2d-detailed-split)

TEST_STATUS:
- 47/47 pytest tests/live2d
- 22/22 TASK-021 integration checklist
- HF Space real images img0..img3 verified (tests/live2d/REAL_IMAGE.md)

## Deliverable summary
- Post-process module: `common/utils/live2d_split.py`, `common/utils/live2d_psd.py`
- CLI: `inference/scripts/live2d_detailed_split.py` + flag on `inference_psd.py`
- Demo: `demo/app.py` checkboxes "Live2D detailed split" / "Show split preview" (OFF = upstream path)
- HF real-image path documented in STATE history / REAL_IMAGE.md

## ENVIRONMENT（復元）
```
W=/mnt/stx  # or /workspace if free; delete /workspace/riichi-city if hydrate fills disk
cd $W && git clone -b live2d-detailed-split https://github.com/nsushiyama/see-through.git
cd see-through && git config user.name nsushiyama && git config user.email 169270041+nsushiyama@users.noreply.github.com
export TMPDIR=$W/tmp; mkdir -p $TMPDIR
python3 -m venv $W/venv && $W/venv/bin/pip install --no-cache-dir \
  numpy==2.2.6 opencv-python-headless==4.13.0.92 Pillow==12.1.1 psd-tools==1.14.2 \
  scipy scikit-image pytest pyyaml pycocotools==2.0.11 gradio_client
PYTHONPATH=common:. $W/venv/bin/python -m pytest tests/live2d -q
```

## HF Space extra deps for Live2D mode
Same as demo/requirements.txt plus: `psd-tools`, `opencv-python-headless`, `scipy`, `scikit-image`
(already listed for the post-process path; no torch/SAM extra required for current splitters).
