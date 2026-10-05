# STATE

STATUS: IN_PROGRESS

CURRENT_PHASE:
final verification

LAST_COMPLETED_TASK:
TASK-020 real-image via HF Space

CURRENT_TASK:
TASK-021 final checks

NEXT_TASK:
(optional) TASK-022 SemanticSam / pose hook

BLOCKERS:
- none
- 注意: box に GPU 無し。ルート FS は他エージェントのデータで頻繁に満杯 → `/mnt/stx` tmpfs + **commit 毎に push**。
- `/workspace/riichi-city` が hydrate で復帰したら削除して空きを確保。
- torch 未インストール → inference_utils.py は import 不可。新モジュールは torch 非依存。

LAST_GOOD_COMMIT:
(see git log; TASK-020)

TEST_STATUS:
47 passed (live2d suite) + real HF img0..img3 verified (see tests/live2d/REAL_IMAGE.md)

## HF Space 実画像ルート
- Space: `24yearsold/see-through-demo`
- API: `gradio_client.Client(...).predict(handle_file(img), 768, 42, False, api_name='/inference')`
- 手順: `fetch_hf_space.py` → `psd_to_srcdir.py` → `run_detailed_split` / demo checkbox
- Primary: test_image1（スカート全身）: 17 → 23 → 45 layers; bg cleanup on legwear; all LR/overlap/offset checks OK

## ENVIRONMENT（復元手順）
```
W=/mnt/stx   # or /workspace if free
cd $W && git clone -b live2d-detailed-split https://github.com/nsushiyama/see-through.git
cd see-through && git config user.name nsushiyama && git config user.email 169270041+nsushiyama@users.noreply.github.com
export TMPDIR=$W/tmp; mkdir -p $TMPDIR
python3 -m venv $W/venv && $W/venv/bin/pip install --no-cache-dir numpy==2.2.6 opencv-python-headless==4.13.0.92 Pillow==12.1.1 psd-tools==1.14.2 scipy scikit-image pytest pyyaml pycocotools==2.0.11 gradio_client
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=common:. $W/venv/bin/python -m pytest tests/live2d -q
```
