# STATE

STATUS: IN_PROGRESS

CURRENT_PHASE: analysis / scaffolding

LAST_COMPLETED_TASK:
TASK-001 既存 See-through パイプライン解析（AGENT/ANALYSIS.md）

CURRENT_TASK:
TASK-002 テスト基盤（合成キャラ semantic RGBA 生成器）

NEXT_TASK:
TASK-003 demo/app.py 取り込み

BLOCKERS:
- box に GPU 無し（nvidia-smi 不在）、RAM 16GB（空き約3GB）、ルート FS 満杯 → 実 See-through 推論（SDXL LayerDiff3D + Marigold）は box で実行不可。
  後処理は合成データ + 純粋ユニットテストで検証。実画像テストは TASK-020 で別手段（HF Space 経由等）を検討。
- torch 未インストール（入れる余地なし）→ inference_utils.py を import できない。新モジュールは torch 非依存で書く。

LAST_GOOD_COMMIT:
a25a549 (upstream main; 未改変)

TEST_STATUS:
まだテスト無し

## ENVIRONMENT（次の Agent 向け復元手順）
- 作業ツリー: `/mnt/stx/see-through`（tmpfs。消えていたら下記で再作成）
  ```
  sudo mkdir -p /mnt/stx && mountpoint -q /mnt/stx || sudo mount -t tmpfs -o size=3G,mode=1777 tmpfs /mnt/stx
  cd /mnt/stx && git clone -b live2d-detailed-split https://github.com/nsushiyama/see-through.git
  cd see-through && git config user.name nsushiyama && git config user.email 169270041+nsushiyama@users.noreply.github.com
  export TMPDIR=/mnt/stx/tmp; mkdir -p $TMPDIR
  python3 -m venv /mnt/stx/venv && /mnt/stx/venv/bin/pip install --no-cache-dir numpy==2.2.6 opencv-python-headless==4.13.0.92 Pillow==12.1.1 psd-tools==1.14.2 scipy scikit-image pytest
  ```
- テスト: `cd /mnt/stx/see-through && /mnt/stx/venv/bin/python -m pytest tests/live2d -q`
- gh は `nsushiyama` で認証済み。push 先は origin（fork）のみ。upstream には触れない。
