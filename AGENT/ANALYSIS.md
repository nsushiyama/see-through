# TASK-001 既存 See-through パイプライン解析（実コード確認済み）

調査対象: fork `nsushiyama/see-through` @ upstream `a25a549`（2026-10-05 clone）
および HF Space `24yearsold/see-through-demo` @ `9cc6ee4b`（app.py / requirements.txt / README.md を取得して確認）。
Space の `common/utils/inference_utils.py` と `common/utils/io_utils.py` は GitHub 版と **完全一致**（diff 0）。

## 1. semantic layer 生成
- `common/utils/inference_utils.py::apply_layerdiff(imgp, pretrained, ..., resolution=1280)`
  - 入力 RGBA を `common/utils/cv.py::center_square_pad_resize(img, resolution, return_pad_info=True)` で
    **中央正方パディング＋resolution へリサイズ**し `src_img.png` として保存（＝以降の全レイヤーの座標系）。
    pad 規則: `sz=max(h,w); px1=(sz-w)//2; py1=(sz-h)//2` → `sz×sz` を `resolution×resolution` へ。
  - tag_version `v3`（デフォルト checkpoint `layerdifforg/seethroughv0.0.2_layerdiff3d`）:
    - body pass (`group_index=0`): `['front hair','back hair','head','neck','neckwear','topwear','handwear','bottomwear','legwear','footwear','tail','wings','objects']`
    - head pass (`group_index=1`): head レイヤーの bbox から元画像を `_crop_head` で切り出し再推論 →
      `['headwear','face','irides','eyebrow','eyewhite','eyelash','eyewear','ears','earwear','nose','mouth']`
      を **resolution×resolution のフルキャンバスに貼り戻して** 保存。
    - 各 tag は `<save_dir>/<srcname>/<tag>.png`（フルキャンバス RGBA、隠れ部分補完済み）。
  - tag_version `v2`: `VALID_BODY_PARTS_V2`（hair, eyes 等 19 tag）。
- 腕・脚の素肌専用レイヤーは v3 に無い（素肌腕は handwear/topwear、脚は legwear 等に含まれる想定）→ 腕/脚分割は handwear/legwear/topwear を対象にする。

## 2. RGBA レイヤー保存
- `apply_layerdiff` 内 `Image.fromarray(rst).save(<tag>.png)`（フルキャンバス）。
- `apply_marigold(...)` が各 tag の `<tag>_depth.png`、`info.json`（`{'parts': {tag: {}}}`）、`reconstruction.png` を保存。
  `compose_list={'eyes':[eyewhite,irides,eyelash,eyebrow],'hair':[back hair,front hair]}` で合成して深度推定し、サブタグ毎に depth を保存。
- `further_extr` の読み込み: `common/utils/io_utils.py::load_parts(srcd)` → `load_part(<tag>.png)`:
  alpha>10 の bbox で **crop** し `{'img','depth'(float 0-1),'xyxy','mask','tag'}` を返す（空レイヤーは None）。
- `inference_utils.py::save_part(tag, saved, part_dict, crop=True, save_to_disk=...)`: 再 crop、depth_median 計算。

## 3. 左右分割
- `inference_utils.py::further_extr(srcd, rotate, save_to_psd, tblr_split)` 内:
  - `tag_lr_split(tag, tag2pinfo)` → `part_lr_split(tag, part_info)` → `cv2.connectedComponentsWithStats(connectivity=8)`
    上位2成分を `label_lr_split` で x 中心比較、`process_cuts` で成分 bbox に crop（1px dilate した成分マスクで alpha を乗算）。
  - **命名: 画面左の成分 → `<tag>-r`、画面右 → `<tag>-l`**（＝キャラクター自身の左右。正面向き前提）。
  - 3番目以降の小成分は捨てられる（既知挙動）。
  - 対象: `handwear`, `eyewhite`, `irides`, `eyelash`, `eyebrow`, `ears`（tblr_split=True 時）。
  - v2 の `eyes`（CC>=5 で eyel/eyer/browl/browr）と `hair`（`torchcv.cluster_inpaint_part` で深度 KMeans 2分割 → hairf/hairb、LaMa inpaint）も同関数内。v3 では `hair`/`eyes` tag は存在しないため実行されない。
  - 後処理 CLI: `inference/scripts/heuristic_partseg.py seg_wdepth|seg_wlr`（→ `seg_wdepth_psd`, `seg_wlr_psd`, `psd2partdicts`）。
- Web デモ UI 上は「Split left/right arms & legs」チェック = `tblr_split`。

## 4. depth / further extraction
- depth: `apply_marigold`（`modules/marigold/MarigoldDepthPipeline`、`24yearsold/seethroughv0.0.1_marigold`）。
- further extraction: `further_extr`（上記）。nose/mouth は `fullpage` の RGB で上書き（**可視画素復元の前例**）。
  face 基準で nose/mouth/eyes と ears の depth_median を調整。
- `common/utils/torchcv.py::cluster_inpaint_part(depth, mask, img, inpaint='lama')`: 深度 KMeans で層分割＋inpaint。

## 5. PSD 生成
- `inference_utils.py::dump_parts_psd(tag2pinfo, frame_size, psd_savep)`: depth_median 降順（奥→手前）に並べ
  `common/utils/io_utils.py::save_psd(savep, img_list, h, w, mode='RGBA')` で `<srcd>.psd` と `<srcd>_depth.psd`、`<srcd>.psd.json`。
- `save_psd`: `psd_tools.PSDImage.new(mode, size=(h, w), depth=8)` → `create_pixel_layer(img, name=tag, top=y1, left=x1)`。
  **注意: PIL 規約では size=(width,height) なので (h,w) は非正方で転置バグになる**（現状は src_img が常に正方なので顕在化しない）。
  → 新規コードでは `size=(W,H)` を正しく渡す。既存関数は変更しない。
- ライブラリ: `psd-tools==1.14.2`（Space requirements）。`PSDImage.create_group(layer_list, name)` / `Group.new` が存在 → **グループ作成可能**（TASK-008 で round-trip 検証）。
- PSD キャンバス = `src_img.png` = 正方パディング済み resolution×resolution（元画像サイズではない）。

## 6. Gradio UI
- GitHub リポジトリには Gradio デモは**無い**（`ui/` は PyQt の Live2D アノテーション UI）。
- HF Space `24yearsold/see-through-demo` の `app.py`（license: MIT, Space README の frontmatter）:
  - `inference(image, resolution=768, seed=42, tblr_split=False)`（`@spaces.GPU(duration=120)`）:
    `apply_layerdiff` → `apply_marigold` → `_collect_layer_gallery(saved)` → `further_extr(saved, save_to_psd=True, tblr_split)` → PSD を返す。
  - `gr.Blocks`: Image, Resolution slider(768-1280), Seed, Checkbox「Split left/right arms & legs」, Run → File(PSD) + Gallery。
  - モデルは CPU にプリロードし `_move_to_gpu()` で `inference_utils.layerdiff_pipeline/marigold_pipeline` に注入。
- → fork の `demo/app.py` に attribution 付きでコピーし改造する（TASK-003）。

## 7. SAM / SAM2 / body parsing / landmark
- `common/modules/sam/*`（SAM / SAM-HQ 実装, `build_sam.py`, `predictor.py`, `automatic_mask_generator.py`）。
- `common/modules/semanticsam.py::SemanticSam(class_num=19, model_type='h_hq')`（`extend_sam.py::BaseExtendSam` 継承）, `.inference(rgb)` → 19クラス mask。
  - HF ckpt `24yearsold/l2d_sam_iter2` / `checkpoint-18000.pt`（README「SAM Body Parsing」）。
  - 使用箇所: `inference/scripts/parse_live2d.py::sam_infer_l2d`, `facedet_sam`, `inference/scripts/infer_sam.py::sam_parse_body_samples`, `training/train/train_partseg.py`。
  - クラスは `common/live2d/scrap_model.py::VALID_BODY_PARTS_V1/V2` 相当（hair, face, eyes, topwear, handwear, legwear …）。
    **上腕/前腕/太腿/すね・髪房の区別は無い** → 腕・脚・髪の細分化には直接使えない（semantic 境界の補助のみ）。
- SAM2: `annotators/lang_sam`（GroundingDINO+SAM2, `requirements-inference-sam2.txt`）。`parse_live2d.py::infer_langsam`。
- landmark/pose: `annotators/anime_face_detector`（mmdet/mmpose 顔ランドマーク）, `annotators/bizarre_tagger/pos_estimator.py`（detectron2 ベース体 keypoint）。いずれも mmcv/detectron2 等の重い依存で Space の requirements に無い。
- 深度クラスタ: `torchcv.cluster_inpaint_part`。

## 結論・設計判断
- 細分化は **semantic RGBA（フルキャンバス）に対する純粋な後処理**として新モジュールで実装し、既存関数は一切変更しない。
- 腕/脚/髪/袖の分割 mask は、既存モデルに該当クラスが無いため **形状解析（距離変換・骨格・主軸・幅プロファイル）** を既定とし、
  `SemanticSam`/pose 等は optional な mask provider フックとして後から差し込める構造にする（新規巨大依存は追加しない）。
- 左右・目・眉・耳・靴・独立装飾は CC（既存 `part_lr_split` と同じ規則: 画面左 = キャラ右）。
- 座標: 内部は `src_img.png` 座標（resolution 正方）フルキャンバス。元画像が与えられた場合は pad 情報を再計算して
  **元画像キャンバス（例 1024×1536）へ逆変換**して出力（全レイヤー同一サイズ・offset 0）。

# 変更計画（どのファイルをどう変えるか）
| 種別 | ファイル | 内容 |
|---|---|---|
| 追加 | `common/utils/live2d_split.py` | torch 非依存の後処理: フルキャンバス化、LR分割、目/眉/耳/靴 CC、髪房、顔/首、腕/脚（関節オーバーラップ）、袖、スカート、装飾、可視画素復元、フォールバック、段階ログ、preview |
| 追加 | `common/utils/live2d_psd.py` | グループ付きフルキャンバス PSD 書き出し（psd-tools create_group、失敗時 prefix 名） |
| 追加 | `inference/scripts/live2d_detailed_split.py` | 既存出力ディレクトリに対する CLI（GPU不要） |
| 変更 | `inference/scripts/inference_psd.py` | `--live2d_detailed_split`, `--split_preview` 追加（OFF 時コード経路不変） |
| 追加 | `demo/app.py`, `demo/README.md`, `demo/requirements.txt` | HF Space app.py（MIT, attribution）をコピーし「Live2D detailed split」「Show split preview」追加。OFF 時は既存処理のみ |
| 追加 | `tests/live2d/` | 合成キャラ RGBA レイヤー生成器 + pytest（キャンバス一致・offset・alpha・再合成・L/R・overlap・fallback・OFF 不変） |
