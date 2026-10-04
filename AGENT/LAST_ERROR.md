# LAST_ERROR

（現在、未解決の重大エラーなし）

## 過去の環境問題（解決済み・参考）
- 2026-10-05: box のルート FS が 100% 使用（`/workspace` に書けない、他エージェントのデータのため削除しない）。
  `/dev/shm` は noexec のため venv の .so が `failed to map segment` でロード不可。
  → 対処: `sudo mount -t tmpfs -o size=3G,mode=1777 tmpfs /mnt/stx` に作業ツリーと venv を置く（揮発。永続は git push）。
