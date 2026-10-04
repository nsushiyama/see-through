# WORKLOG（新しいものを下に追記）

## 2026-10-05 TASK-000 / TASK-001
- fork 作成: https://github.com/nsushiyama/see-through 、branch `live2d-detailed-split`、remote `upstream` = shitagaki-lab/see-through（push しない）。
- 環境: GPU 無し、RAM 16GB、ルート FS 満杯 → /mnt/stx に tmpfs を mount して作業（STATE.md ENVIRONMENT 参照）。
- HF Space `24yearsold/see-through-demo`（MIT）の app.py/requirements.txt/README.md を取得。inference_utils.py/io_utils.py は GitHub 版と同一。
- 実コードを読み、AGENT/ANALYSIS.md に 1〜7 の所在（関数・パス）と変更計画を記録。TODO.md に TASK-002〜022 を作成。
- 重要発見: (a) 既存 LR 分割の命名は「画面左成分 → -r（キャラ右）」, (b) 全レイヤーは src_img.png（正方パディング＋resolution リサイズ）座標,
  (c) save_psd は size=(h,w) で非正方時に転置する潜在バグ（既存は正方なので無害）, (d) 既存 body parsing SAM は 19 クラスで上腕/前腕等の区別無し,
  (e) psd-tools 1.14.2 に create_group あり。

## 2026-10-05 TASK-002
- tests/live2d/synth.py: v3 tag 名で See-through 出力形式の合成全身キャラ（隠れ部分あり、非正方元画像 768x1024 → 正方パディング）。gt.json に関節。
- tests/live2d/run.sh（TMPDIR=/mnt/stx/tmp）。2 passed。
- 発見: load_part は下端/右端 10% 帯だけのレイヤー（靴）を捨てる既存挙動 → ANALYSIS 追記。
- 一時的に box の Shell が spawn 失敗（数分で回復）。原因は /tmp 満杯 or メモリ逼迫と推定。TMPDIR 必須。
- ユーザー指示: 実画像テストは HF Space 経由で行う（GPU 不在を BLOCKED 扱いにしない）→ TASK-020a を前倒し。

## 2026-10-05 TASK-003 / TASK-020a
- demo/app.py: Space app.py を MIT attribution 付きで取り込み（_root と Examples ガードのみ変更, 原本 demo/app_space_original.py）。7bbd7ef。
- tests/live2d/fetch_hf_space.py, psd_to_srcdir.py。test_image1 で Space 推論成功（110.9s）→ 所見は STATE.md。
- box 更新騒ぎ: 一時的に /mnt/stx が見えなくなったが実際は残っていた（7bbd7ef と WIP を回収して push）。ルート FS は再び 100%。
- common/utils/live2d_split.py の WIP（未テスト）は次の TASK-004 commit でテストと一緒に入れる。

## TASK-004
- common/utils/live2d_split.py: Part, load_semantic_layers（crop しない・下端レイヤーも保持）, overlap_px（長辺1-3%クランプ）,
  pad_info/to_original_canvas/from_original_canvas（premultiplied resize）, composite, validate_split/safe_split（失敗時は元レイヤー保持）。
  同ファイルに LR 分割・limb 解析ヘルパ（未テスト, TASK-005/010 でテスト）。
- tests/live2d/test_core.py 6件追加 → 8 passed。
