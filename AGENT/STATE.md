# STATE

STATUS: IN_PROGRESS

CURRENT_PHASE:
core post-processing module

LAST_COMPLETED_TASK:
TASK-017 可視画素復元（ソース単位の前方透過率で可視判定, 内部は元画像 RGB, 境界 feather=長辺0.4%, alpha 不変, 隠れ画素は不変）

CURRENT_TASK:
TASK-018 split preview（色分け mask + レイヤー数 + 名前一覧）

NEXT_TASK:
TASK-019 demo/app.py UI チェックボックス

BLOCKERS:
- none（実画像テストは HF Space 経由で実施可能。GPU 不在は BLOCKER ではない）
- 注意: box に GPU 無し。ルート FS は他エージェントのデータで頻繁に 100% 満杯になる（/workspace に書けないことがある）。
  その場合は exec 可能な tmpfs `/mnt/stx` を使い、**commit 毎に必ず push**（tmpfs は box 更新で消える可能性あり）。
- torch 未インストール → inference_utils.py は import 不可。新モジュールは torch 非依存。

LAST_GOOD_COMMIT:
3d44105 (tests 2 passed)。7bbd7ef = demo 取り込み（構文確認のみ）

TEST_STATUS:
tests/live2d: 37 passed。実データ img0: 再合成の元画像との平均絶対誤差 12.24 → 4.21（restore 後）、目視で外観一致

## HF Space 実画像ルート
- Space: `24yearsold/see-through-demo`（https://24yearsold-see-through-demo.hf.space, ZeroGPU, 匿名利用可, 1回約110秒 @768）
- API: `gradio_client.Client('24yearsold/see-through-demo').predict(handle_file(img), 768, 42, False, api_name='/inference')`
  引数 = (image filepath, resolution, seed, tblr_split) → `(psd_filepath, gallery: list[{image: .webp(lossy), caption: tag}])`
- 手順: `python tests/live2d/fetch_hf_space.py IMG OUTDIR` → `python tests/live2d/psd_to_srcdir.py OUTDIR`
  （PSD レイヤーを full canvas の <tag>.png に戻し、src_img=入力の正方パディング、depth=PSD 重ね順、info.json を生成）
- テスト画像: `https://huggingface.co/spaces/24yearsold/see-through-demo/resolve/main/common/assets/test_image{,1,2,3,4}.png`（928x1232 RGB）。
  test_image1（スカート・全身正面）で取得成功。
- 実データ所見（test_image1 @768, PSD 17 レイヤー; headwear/objects/tail/wings/eyewear は空）:
  - **handwear = 両腕（素肌＋パフスリーブ込み、袖の下も補完）** の2連結成分 → 腕分割と袖分割（布/肌を色で分離）は handwear で行う。
  - **legwear = 素肌の脚 + 白背景がグレー不透明領域として混入（約40万px, モデル側の失敗）** → 巨大レイヤーは fallback か脚領域抽出が必要。
  - topwear は胴体のみ（オフショルダー）。bottomwear はスカート。

## ENVIRONMENT（復元手順）
```
# 作業場所: 空きがあれば /workspace、無ければ /mnt/stx（tmpfs, exec 可）
df -h /     # 空き確認
W=/workspace   # or: sudo mkdir -p /mnt/stx && (mountpoint -q /mnt/stx || sudo mount -t tmpfs -o size=3G,mode=1777 tmpfs /mnt/stx); W=/mnt/stx
cd $W && git clone -b live2d-detailed-split https://github.com/nsushiyama/see-through.git
cd see-through && git config user.name nsushiyama && git config user.email 169270041+nsushiyama@users.noreply.github.com
export TMPDIR=$W/tmp; mkdir -p $TMPDIR
python3 -m venv $W/venv && $W/venv/bin/pip install --no-cache-dir numpy==2.2.6 opencv-python-headless==4.13.0.92 Pillow==12.1.1 psd-tools==1.14.2 scipy scikit-image pytest pyyaml pycocotools==2.0.11 gradio_client
```
- /dev/shm は noexec のため venv 不可。
- テスト: `PY=$W/venv/bin/python bash tests/live2d/run.sh`
- gh は `nsushiyama` で認証済みが前提（`gh auth setup-git`）。push 先は origin（fork）のみ。upstream / PR には触れない。
