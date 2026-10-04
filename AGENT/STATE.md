# STATE

STATUS: IN_PROGRESS

CURRENT_PHASE: analysis / scaffolding

LAST_COMPLETED_TASK:
TASK-002 テスト基盤（tests/live2d/synth.py, run.sh）

CURRENT_TASK:
TASK-020a HF Space 経由で実画像の See-through 出力を取得

NEXT_TASK:
TASK-003 demo/app.py 取り込み

BLOCKERS:
- box に GPU 無し・RAM 空き約3GB・ルート FS 100% 満杯 → box では See-through 推論しない。**実画像テストは公式 HF Space `24yearsold/see-through-demo` を gradio_client で呼び出して実行する（ユーザー指示, BLOCKED ではない）**。取得した semantic RGBA レイヤー/通常 PSD を detailed split に入力して検証。
- 全ての作業ファイル・venv・一時ファイルは /mnt/stx（tmpfs）に置く。TMPDIR=/mnt/stx/tmp を必ず設定（/tmp も満杯）。
- torch 未インストール → inference_utils.py は import 不可。新モジュールは torch 非依存。

LAST_GOOD_COMMIT:
a25a549 (upstream main; 未改変)

TEST_STATUS:
tests/live2d: 2 passed (synth layout, existing load_parts compat)

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
