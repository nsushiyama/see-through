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

## TASK-005/006/007
- split_lr_cc（既存 part_lr_split と同規則: 上位2成分, 画面左=キャラ右 _R。小成分/薄い縁は近い側へ割当て欠損なし）。
- stage_lr_extra: footwear/earwear CC、legwear は CC→無理なら体中心線（neck/face 中央値 x）で分割、canvas の35%超は背景混入とみなし保持。
- test_lr.py 4件。実データ img1: original 17 → after LR 23 → extra 24、全左右正しい（_R が画面左）。

## TASK-008
- live2d_psd.save_live2d_psd: PSDImage.new(size=(W,H)) で正しい向き、全レイヤー full canvas/offset 0、
  グループは z-order 上の連続区間ごと（同名再出現は Hair_2 等）→ 描画順を崩さない。失敗時 "<Group>_name" フラット。
- read_psd_layers（テスト用）。test_psd.py 2件、14 passed。

## TASK-009
- run_detailed_split(srcd, original, out_psd, overlap_ratio, restore_visible, preview, use_groups):
  load → LR → stage_lr_extra → DETAILED_STAGES（後続タスクで登録, 各 stage 例外は握り潰して不変）→ 元画像キャンバスへ → depth 安定ソート → (restore/preview は後続) → PSD。
  段階ログ "[live2d] original semantic layers: N / after LR split / after detailed split / final PSD layers"。
- CLI と inference_psd.py フラグ（OFF 時は upstream と同一コード経路）。test_pipeline.py 7件、21 passed。

## TASK-010
- split_arm: neck 基部を近位 anchor に geodesic 距離 → arm_joints（手首=0.68-0.86L 最小幅, 肘=肩-手首中点を屈曲で補正）→
  segment_with_overlap で upper_arm/forearm/hand（各関節 ±ov=長辺2%）。長さ<長辺8% なら hand のみ。
- 袖布: 手の (a,b) 中央値からの彩度距離 > max(10, 3*手の広がり) を布とする（明度は無視→影の誤検出防止）。
  1回目（Lab 2-means, ΔE>22）は実データ左腕の影を袖と誤判定 → 方式変更（同じ方法2回失敗ルール前に切替）。
- 実データ img1: パフ袖は肌とほぼ同色で分離されず upper_arm に含まれる（既知制限）。test_arms.py 4件。

## TASK-011
- split_topwear: 各 topwear 画素を「胴体中心線/胴体半幅」「腕中心線/(1.6*腕半幅)」の正規化距離最小で torso/sleeve に割当て。
  腕シルエット(0.6*腕半幅 dilate)外は torso 固定。袖成立条件: 腕との重なり≥60%, 腕方向の広がり≥肘距離の40%。
  肩側に ov 幅の重複を付与、肘で upper/lower（lower が小さければ1枚）、先端15%帯の色差>10 で cuff。
- 実データ: 初回はブラウス側部を袖として lower_sleeve まで細切れ → 小 lower 統合で torso+upper_sleeve_R/L に。test_topwear.py 1件。

## TASK-012
- split_leg: 脚上端を anchor に geodesic、leg_joints（膝=股-足首中点を屈曲補正、footwear 無しなら足首=0.8-0.94L 最小幅で foot も）。test_legs.py 2件。

## TASK-013
- 髪は CC 非依存: 前髪は額上方ピボットからの角度セクター（質量分位で初期境界→房の隙間=最大半径の谷へスナップ）、
  幅が顔幅の60%未満なら3分割。サイドは顔幅外かつ目の高さ以下（ov/2 で前髪へ重複）。後ろ髪は中心帯(顔半幅の35%)+左右角度セクター（01=下方）。
- 失敗1: 顔幅を3-97%分位で取り外側の房がサイド扱い→ 0.5-99.5% に。失敗2: サイドの列方向上方拡張が前髪房を侵食→拡張削除。
- test_hair.py 3件（単一連結の前髪でも房先端がほぼ1パーツに属する、左右順序、顔無し fallback）。
- 既知制限: サイド髪は目の高さで水平に切れる（重複で隙間は防止）。
