# Domain Adaptation (DA) 学習ノート（補足資料）

本資料は `run/train_maestro_ddp_DA.py`（内部で `my_mt3/train_DA.py::train_loop_distributed_DA` を呼び出し）に関する補足です。DC（Domain Confusion）とPseudo それぞれの有無による分岐、loss の定義、optimizer の更新順、I/O 仕様などをまとめます。

## バッチ構造と入出力

- synth（教師あり）:
  - `mels_s`: [Nchunks, T(=input_frames), F(=n_mels)]（例: [Nchunks, 256, 256]）
  - `y_in_s`, `y_tg_s`: [Nchunks, S]（各チャンクのトークン列。BOS/シフト済み）
  - 処理: `mem_s = enc(mels_s)` → `logits_s = dec(y_in_s, mem_s)`

- real（未ラベル）:
  - `mels_r`: [Nchunks, T, F]（`use_dc` または `use_pseudo` のときのみ使用）
  - 処理: `mem_r = enc(mels_r)`（DC と Pseudo の材料）

## 学習ループの全体構造（分岐別）

記号: enc=エンコーダ, dec=デコーダ, D=識別器, EMA=teacher

### A) use_dc=False, use_pseudo=False（教師ありのみ）
```
for each synth batch:
  mem_s = enc(mels_s)
  sup = CE(dec(y_in_s, mem_s), y_tg_s)
  total = sup
  update(opt_t)
```

### B) use_dc=True, use_pseudo=False（DC のみ）
```
for each synth+real batch:
  mem_s = enc(mels_s)
  mem_r = enc(mels_r)

  # Disc step
  disc = BCE(D(mem_s), 0) + BCE(D(mem_r), 1)
  update(opt_c)

  freeze(D)
  sup = CE(dec(y_in_s, mem_s), y_tg_s)
  adv = BCE(D(mem_s), 0.5) + BCE(D(mem_r), 0.5)
  total = sup + λ*adv
  update(opt_t)
  unfreeze(D)
```

### C) use_dc=False, use_pseudo=True（Pseudo のみ）
```
for each synth+real batch:
  mem_s = enc(mels_s)
  sup = CE(dec(y_in_s, mem_s), y_tg_s)

  if epoch >= pseudo_start:
    y_pseudo, pmax, margin = teacher(greedy_with_probs(mels_r))
    build y_in_p, y_tg_p (BOS 除外)
    token_mask = note-based 重要/不要スコアから作成
    y_tg_masked = apply_mask(y_tg_p, token_mask)
    logits_r = dec(y_in_p, enc(mels_r))
    unsup = CE(logits_r, y_tg_masked)
  else:
    unsup = 0

  total = sup + w*unsup
  update(opt_t)
  EMA update
```

### D) use_dc=True, use_pseudo=True（DC + Pseudo）
```
for each synth+real batch:
  mem_s = enc(mels_s)
  mem_r = enc(mels_r)

  # Disc step
  disc = BCE(D(mem_s), 0) + BCE(D(mem_r), 1)
  update(opt_c)

  freeze(D)
  sup = CE(dec(y_in_s, mem_s), y_tg_s)
  adv = BCE(D(mem_s), 0.5) + BCE(D(mem_r), 0.5)

  if epoch >= pseudo_start:
    y_pseudo, pmax, margin = teacher(greedy_with_probs(mels_r))
    token_mask = note-based
    y_tg_masked = apply_mask(y_tg_p, token_mask)
    logits_r = dec(y_in_p, enc(mels_r))
    unsup = CE(logits_r, y_tg_masked)
  else:
    unsup = 0

  total = sup + λ*adv + w*unsup
  update(opt_t)
  unfreeze(D)
  EMA update
```

## Loss の構成式

- 教師あり CE（synth）:
  - `sup = CE(dec(y_in_s, enc(mels_s)), y_tg_s)`
- 敵対的（adv, synth+real）:
  - `adv = BCE(D(enc(mels_s)), 0.5) + BCE(D(enc(mels_r)), 0.5)`
- 識別器（disc, 真偽判別）:
  - `disc = BCE(D(enc(mels_s)), 0) + BCE(D(enc(mels_r)), 1)`
- Pseudo CE（real, BOS 除外 + token mask 適用）:
  - `unsup = CE(dec(y_in_p, enc(mels_r)), y_tg_masked)`
- 合計:
  - `total = sup + λ·adv + w·unsup`（不要な項は 0）

## Optimizer の更新順

- A: `[opt_t]`（`total = sup`）
- B: `[opt_c] → [opt_t]`（`total = sup + λ·adv`）
- C: `[opt_t] → [EMA update]`（`total = sup + w·unsup`）
- D: `[opt_c] → [opt_t] → [EMA update]`（`total = sup + λ·adv + w·unsup`）

## chunk とバッチ

- データセットは 1曲 → 複数チャンクを生成（train はランダム K 個、val/test はスライド走査）
- collate でチャンクをフラット化し、「1チャンク=1サンプル」としてモデルに投入
- モデル I/O: `enc([B,T,F]) → mem`、`dec([B,S], mem) → logits`

## キャッシュ戦略

- `--cache-root` 配下に「データセット名/用途（synth|real）」で自動分岐
  - 例: `cache/wave_sr16000/GiantMIDI-PIano/synth` と `cache/wave_sr16000/maestro-v3.0.0/real`
- 事前プリフェッチ（`--prefetch_cache_workers > 0`）で WAV→NPY を並列生成
- 生成後、学習には `.npy` パスへ差し替え（`use_cache=False` で二重化回避）
- 競合対策: `ensure_wave_cache` はユニーク tmp + `os.replace` で原子的に作成

## 主なCLIスイッチ（抜粋）

- DC（Domain Confusion）
  - `--use_dc`（ONで有効）
  - `--lambda_adv`, `--lr_t`, `--lr_c`, `--chunk_frames`（未指定なら約 0.1s を自動）
- Pseudo
  - `--use_pseudo`（ONで有効）
  - `--pseudo_start_epoch`, `--ema_decay`, `--unsup_weight`, `--pseudo_max_len`
  - `--top_frac`, `--bot_frac`（note-based token mask の選択範囲）
- キャッシュ
  - `--cache-root`, `--cache-dir-synth`, `--cache-dir-real`, `--prefetch_cache_workers`, `--no-cache`

## 注意点 / トラブル対策

- DDP の GPU マッピング
  - 可視化 GPU 枚数と `--nproc_per_node` を一致
  - 単 GPU の場合は `--nproc_per_node=1`
- NCCL 初期化で待つ/ハング
  - `LOCAL_RANK` に応じて `torch.cuda.set_device(local_rank)` をセット
  - 単 GPU なら `gloo` を検討
- キャッシュ競合
  - 本実装は `tempfile.mkstemp` + `os.replace` で並列耐性あり
  - それでも不安なら `--prefetch_cache_workers` で事前温め推奨

---

以上。必要に応じて本資料を更新してください。*** End of document. ***
