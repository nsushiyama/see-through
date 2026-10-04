# MASTER_GOAL

See-through（https://github.com/shitagaki-lab/see-through ／ HF Space `24yearsold/see-through-demo`）の
Webデモ／GitHub実装を改造し、通常の約23レイヤー出力に加えて
**「Live2D / Spine 向け 50〜80 パーツ程度の細分化PSD」** を出力できるモードを実装する。

作業リポジトリ: fork `nsushiyama/see-through`、ブランチ `live2d-detailed-split`
（upstream `shitagaki-lab/see-through` へは push / PR しない）

---

## 0. 起動プロトコル（毎回必ず実行）

1. `AGENT/MASTER_GOAL.md`（本ファイル）を読む
2. `AGENT/STATE.md`
3. `AGENT/TODO.md`
4. `AGENT/WORKLOG.md`
5. `AGENT/LAST_ERROR.md`
6. `git status`
7. `git log --oneline -20`
8. （環境）`AGENT/STATE.md` の ENVIRONMENT 節に従ってテスト環境を復元する

- 以前の会話履歴に依存しない。状態はファイルと Git から復元する。
- MASTER_GOAL 達成のための「次に実行可能な最小タスク」を TODO.md から自分で選ぶ。
- 1タスク = 調査 → 実装 → テスト → 修正 → commit の短い単位。巨大な作業にしない。
- 実装前に必ず現在のコードを読む。存在しない関数・ファイル・APIを推測で使わない。既存実装を最大限再利用する。
- 各タスク完了時: (1) テスト (2) 結果確認 (3) git commit (4) STATE.md 更新 (5) TODO.md 更新 (6) WORKLOG.md 記録。
  さらに `git push origin live2d-detailed-split`。
- 重大な失敗時は `AGENT/LAST_ERROR.md` に: 何をしていたか / 何が失敗したか / エラーメッセージ /
  試したこと / 次に試すべきこと / 最後に動作した commit を書く。次回起動時はこれを読んで復旧する。
- 同じ方法で2回失敗 → 別の方法を検討。3回失敗 → `BLOCKED` として記録し、別ルートで MASTER_GOAL を目指す。
- Git を checkpoint として使う。正常動作を壊したら LAST_GOOD_COMMIT から原因を追えるようにする。
  未確認の大量変更を一度に commit しない。
- 既存機能を壊さないことを優先。**See-through 通常モードは常に動作する状態を維持する。**
- 実行終了前に「今この瞬間セッションが完全消失しても次のAgentが続行できるか」を確認し、
  不足情報は STATE.md / WORKLOG.md に記録する。
- MASTER_GOAL 未達成の間は人間の追加指示を待って停止しない（承認が本当に必要な外部操作を除く）。
- 完了判定はコードが書けたことではない。以下を全て実施して合格した後にのみ STATE.md を `STATUS: COMPLETE` にする:
  既存モードテスト / Live2D detailed split テスト / PSD生成テスト / レイヤー数確認 / 座標一致確認 /
  透明度確認 / 左右確認 / フォールバック確認。

---

## 1. 基本方針（必須制約）

- See-through 本体の生成モデルは**再学習しない**。
- 既存の約23 semantic layer 生成処理は**変更しない**。23パーツ生成後の**後処理**として細分化する。
- 元画像と同じキャンバスサイズ・座標系を**絶対に維持**する。
- 各パーツは別レイヤーとして PSD に保存する。
- 見えていない部分は See-through が補完した画像をそのまま利用する。
- 単純な矩形 crop で位置情報を失わない。生成済み RGBA の透明部分も維持する。
- 既存の通常 PSD 出力も残す。

## 2. UI

- 現在の Web デモ（Gradio）に **「Live2D detailed split」** チェックボックスを追加。
  - OFF: 既存 See-through と完全に同じ挙動。
  - ON: 通常 See-through 処理 → 既存左右分割 → Live2D 向け追加細分化 → 細分化 PSD 出力。
- デバッグ用 **「Show split preview」**（可能なら）: 元画像の横に各maskの色分けプレビュー、最終レイヤー数、パーツ名一覧。
- ログに段階ごとの枚数: 例 `original semantic layers: 23` / `after LR split: 29` / `after detailed split: 67`。

## 3. 細分化目標

- 50〜80 レイヤー程度（「必ず70枚」ではない）。不自然な水増し禁止。**枚数より意味的な正しさを優先**。
- ベース: See-through の semantic layer（front hair, back hair, head, ears, eyebrow, eyewhite, irides, eyelash,
  topwear, bottomwear, handwear, legwear, footwear, neckwear, headwear, earwear, objects, tail, wings など）。

### 髪
- front hair → front_hair_center, front_hair_left_01, front_hair_left_02, front_hair_right_01, front_hair_right_02, side_hair_left, side_hair_right
- back hair → back_hair_center, back_hair_left_01..03, back_hair_right_01..03
- connected components だけに依存しない。一本の連結領域でも SAM/SAM2・画像特徴・人物ランドマーク・形状解析等で「動かしやすい房」に分割する。

### 顔
- 可能なら face, left_ear, right_ear, neck。

### 目
- left/right_eyebrow, left/right_eyewhite, left/right_iris, left/right_eyelash。
- 左右判定は画面左右ではなく **character_left / character_right**（キャラクター自身の左右）が分かる命名。

### 服
- topwear → torso, left/right_upper_sleeve, left/right_lower_sleeve, left/right_cuff, collar, chest_decoration, waist_decoration など（存在しないものは生成しない）。

### 腕
- 人物ランドマークまたは body parsing で left/right_upper_arm, left/right_forearm, left/right_hand。
- 関節で真っ二つに切らない。肩・肘・手首の境界付近は両パーツに**オーバーラップ**を持たせる
  （upper_arm は肘より少し下まで、forearm は肘より少し上まで）。
- オーバーラップ量は解像度に応じて可変。目安: **長辺の 1〜3%**。

### 脚
- left/right_thigh, left/right_lower_leg, left/right_foot。膝・足首にオーバーラップ。

### スカート等
- bottomwear → 可能なら skirt_front/back/left/right。前後判定できなければ bottomwear_left/center/right 程度でよい。

### 装飾
- リボン・ネクタイ・スカーフ・髪飾りなど独立領域が検出できる場合、main, left_tail, right_tail, center_decoration 等に分割。

## 4. 処理順（必須）

元画像 → See-through → 隠れ部分まで補完された semantic RGBA → 既存左右分割 →
SAM/SAM2/body parsing/landmark 等で細分化用 mask 生成 → semantic RGBA と mask を交差 →
Live2D パーツ作成 → 関節境界をオーバーラップ → PSD 出力

- **元画像そのものを直接 SAM で70分割する方式は禁止**（衣服の下・髪の裏・腕の裏などの隠れ部分が存在しないため）。

## 5. キャンバス・画素

- 全パーツは内部的に**元キャンバスサイズの RGBA**（例: 元画像 1024×1536 なら全パーツ 1024×1536）として保持。小さな crop 画像だけを保存しない。
- PSD 上でも元画像に対して正しい位置にそのまま重なること。
- 可能なら元画像の可視部分はピクセル一致を優先: visible_mask が得られるパーツは `result[visible_mask] = original[visible_mask]` に近い考え方で元画像の可視画素を戻せる構成。
  他パーツとの重なり・境界で不自然になる場合は数 px の feather。

## 6. SAM/SAM2/依存関係

- リポジトリ内既存の SAM/SAM2/body parsing 実装をまず再利用。新しい巨大依存を増やす前に公式コード内の既存機能を確認。不足する場合のみ追加。
- connected components を積極的に使ってよい: 左右の目・眉・耳・靴、独立した髪飾り、独立したリボン。
- CC だけでは不可: 袖と胴体、上腕と前腕、一本につながった髪 → 別の方法を使う。

## 7. フォールバック

- 分割成功 → 細分化パーツへ置換。分割失敗 → **元レイヤーを保持**（例: topwear をそのまま残す。壊れた細分化パーツだけを残さない）。

## 8. PSD 命名・グループ

- 現在の PSD ライブラリで安全にグループを作れるなら Hair / Face / Body / Clothes / Accessory でグループ化。難しければ `Hair_front_hair_L_01` のような prefix。
- 例:
  - Hair: front_hair_center, front_hair_L_01, front_hair_L_02, front_hair_R_01, front_hair_R_02, back_hair_center, back_hair_L_01, back_hair_R_01
  - Face: face, ear_L, ear_R, eyebrow_L, eyebrow_R, eye_white_L, eye_white_R, iris_L, iris_R
  - Body: neck, upper_arm_L, forearm_L, hand_L, upper_arm_R, forearm_R, hand_R, thigh_L, lower_leg_L, foot_L, thigh_R, lower_leg_R, foot_R
  - Clothes: torso, upper_sleeve_L, lower_sleeve_L, cuff_L, upper_sleeve_R, lower_sleeve_R, cuff_R, skirt_center, skirt_L, skirt_R
  - Accessory: ...
  （L/R はキャラクター自身の左右）

## 9. 品質条件

1. 全パーツのキャンバスサイズが元画像と一致
2. PSD 読み込み時に位置ズレなし
3. 透明領域が正しく保持
4. 隠れていた部分は See-through の補完結果が残る
5. 分割後に重ね合わせると元キャラクターの外観が大きく変わらない
6. 左腕・右腕・左脚・右脚を取り違えない
7. 腕・脚の関節部に十分な重複領域
8. 一部の分割に失敗しても PSD 全体が壊れない
9. Live2D detailed split OFF 時は既存 Web デモの結果に影響しない
10. 50〜80 程度の実用的パーツが目標だが、枚数より意味的正しさを優先

## 10. テスト

- 少なくとも1枚の全身アニメキャラクター画像でテスト。
- 確認: 元画像 / 通常 See-through PSD / Live2D detailed split PSD / 最終レイヤー数 / レイヤー名一覧 / mask preview。
- 特に 髪・腕・袖・脚・目 の左右間違い・位置ズレ・欠損。

## 11. 最終報告に含めるもの

変更したファイル / 追加したファイル / 各ファイルの変更内容 / 起動方法 /
Hugging Face Space で必要な追加 dependency / 既知の制限。

**最優先事項: 「See-through の隠れ部分補完能力を維持したまま、同じキャンバス座標上で Live2D 向けに細分化する」こと。**
