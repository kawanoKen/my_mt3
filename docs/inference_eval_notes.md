# 推論 (Inference) / 評価 (Evaluation) ノート（補足資料）

本資料は推論パイプライン (`run/infer_maestro.py`, `my_mt3/infer.py`) と評価パイプライン (`run/eval_maestro.py`, `my_mt3/eval.py`) に関する補足です。

---

## 1. 推論アーキテクチャ

### 1.1 全体フロー

```
WAV → load_audio_mono → waveform
                          ↓
                  LogMelExtractor
                          ↓
              mel [T, F] をチャンク分割
                          ↓
              バッチ化 [B, input_frames, n_mels]
                          ↓
                    Encoder (mel → mem)
                          ↓
              FastDecoderKV (KV Cache decode)
                          ↓
                   token_ids per chunk
                          ↓
            to_midi_from_tokens_piano (MIDI化)
                          ↓
              shift_and_merge_pm (時間シフト＋結合)
                          ↓
                    PrettyMIDI (曲全体)
```

### 1.2 KV Cache によるデコード高速化

デコーダは自己回帰（autoregressive）でトークンを1つずつ生成する。

**旧方式（KV Cache なし）:**
- 毎ステップで全トークン系列 `y=[BOS, t1, t2, ..., tn]` をデコーダに通す
- ステップ n での計算量: O(n) → 全体 O(n^2)

**新方式（KV Cache あり）:**
- `FastDecoderKV` (`my_mt3/decode_kv.py`) を使用
- Self-Attention の K/V を蓄積し、新トークン1つだけを処理
- Cross-Attention の K/V は encoder 出力 (`mem`) から一度だけ事前計算
- ステップ n での計算量: O(1)（attention は O(n) だが行列積は query=1） → 全体 O(n)

**内部構造 (`FastDecoderKV`):**

```
init_cache(mem):
  ├── self_k/v: [L, B, H, max_len, Hd]  ← 空で確保、逐次書き込み
  └── cross_k/v: [L, B, H, Tmem, Hd]    ← mem から一括計算（不変）

forward_step(y_last, cache):   ← y_last: [B, 1]
  ├── embed + positional encoding (step 位置)
  ├── for each layer:
  │   ├── self-attn:  q,k,v = proj(h) → cache 書込 → SDPA(q, k[0:t+1], v[0:t+1])
  │   ├── cross-attn: q = proj(h) → SDPA(q, cross_k, cross_v)
  │   └── FFN
  ├── cache.step += 1
  └── return logits [B, V]
```

**KVCache データ構造:**
- `self_k`, `self_v`: `[L, B, H, MAX, Hd]` — 各レイヤの self-attn 用、in-place 更新
- `cross_k`, `cross_v`: `[L, B, H, Tmem, Hd]` — 各レイヤの cross-attn 用、init_cache で一度だけ計算
- `step`: 現在のデコードステップ（0-indexed）

### 1.3 チャンク分割とマージ

1曲の音声を固定長チャンクに分割して推論し、結果を時間シフトして結合する。

- `need_samples = (input_frames - 1) * hop + n_fft`（1チャンクに必要なサンプル数）
- `stride_samples = stride_frames * hop`（チャンク間のスライド幅）
- 末端が余る場合、最後のチャンクを `total_samples - need_samples` 位置から追加
- 各チャンクの推論結果 MIDI を `t0 = start_sample / sr` だけ時間シフトして結合
- ノートの整列（start, pitch 順）は全チャンク結合後に一括実行

### 1.4 バッチデコード

`_greedy_decode_batch` は複数チャンクを同時にデコードする。

```
mels_bt: [B, T, F]
  → enc → mem: [B, T', D]
  → FastDecoderKV(dec, max_len+1)
  → init_cache(mem)
  → loop:
      logits = forward_step(cur, cache)   # [B, V]
      nxt = argmax(logits)                # [B]
      per-sample EOS check
      cur = nxt.unsqueeze(1)              # [B, 1]
```

- `batch_size` 引数で1回のバッチサイズを制御（デフォルト 16）
- GPU メモリに応じて調整

### 1.5 速度最適化まとめ

| 手法 | 効果 |
|------|------|
| KV Cache (`FastDecoderKV`) | デコード O(n^2) → O(n) |
| `torch.autocast(bfloat16)` | GPU 演算の混合精度 |
| `cudnn.benchmark = True` | 最適カーネル自動選択 |
| `tf32` matmul | float32 行列積の高速化 |
| バッチデコード | 複数チャンクを GPU 並列 |
| ソート最後に一括 | チャンクごとのソートを回避 |
| WAV キャッシュ (`.npy`) | 繰り返し推論時の I/O 削減 |
| CPU スレッド制限 | BLAS/OpenMP の過剰利用を抑制 |

---

## 2. 推論スクリプト

### 2.1 `run/infer_maestro.py`

Piano 専用の推論スクリプト。2つの動作モードを持つ。

**単体ファイルモード:**
```bash
uv run run/infer_maestro.py \
  --ckpt checkpoints/model.pt \
  --root maestro-v3.0.0 \
  --wav path/to/audio.wav \
  --out path/to/output.mid
```

**データセットモード:**
```bash
uv run run/infer_maestro.py \
  --ckpt checkpoints/model.pt \
  --root maestro-v3.0.0 \
  --split validation \
  --out_dir outputs/maestro_val_pred \
  --batch_size 16 \
  --eval
```

**マルチ GPU（手動シャーディング）:**
```bash
CUDA_VISIBLE_DEVICES=0 uv run run/infer_maestro.py --ckpt model.pt --root maestro-v3.0.0 --num_shards 2 --shard_id 0 &
CUDA_VISIBLE_DEVICES=1 uv run run/infer_maestro.py --ckpt model.pt --root maestro-v3.0.0 --num_shards 2 --shard_id 1 &
wait
```

### 2.2 主な CLI オプション

| オプション | デフォルト | 説明 |
|------------|-----------|------|
| `--ckpt` | (必須) | モデル checkpoint パス |
| `--root` | (必須) | MAESTRO v3.0.0 ルートディレクトリ |
| `--split` | `validation` | 対象スプリット |
| `--wav` | - | 単体推論時の入力 WAV |
| `--out` | - | 単体推論時の出力 MIDI |
| `--out_dir` | `outputs/maestro_validation_pred` | データセットモード出力先 |
| `--batch_size` | `16` | チャンクデコードのバッチサイズ |
| `--stride_frames` | `None`（= input_frames） | チャンクスライド間隔 |
| `--max_len` | `1024` | 最大生成トークン数 |
| `--use_cache` | off | WAV→NPY キャッシュを使用 |
| `--num_shards` / `--shard_id` | `1` / `0` | 手動シャーディング設定 |
| `--cpu_threads` | `4` | PyTorch CPU スレッド数制限 |
| `--eval` | off | 推論後に自動評価を実行 |
| `--onset_tolerance` | `0.05` | 評価時の onset 許容誤差 (秒) |
| `--offset_ratio` | `0.2` | 評価時の offset ratio |

### 2.3 `my_mt3/infer.py`（汎用推論関数）

| 関数 | 説明 |
|------|------|
| `greedy_decode(model, mel, ...)` | 単一サンプル greedy decode（KV Cache 使用） |
| `greedy_decode_with_probs(model, mel, ...)` | バッチ版、確信度 (pmax, margin) 付き |
| `to_midi_from_tokens(token_ids, ...)` | 汎用トークン→MIDI 変換（固定長ノート） |
| `to_midi_from_tokens_piano(token_ids, ...)` | Piano 用、Note On/Off 対応 |

---

## 3. 評価パイプライン

### 3.1 評価指標

`mir_eval` ライブラリを使用した標準的なピアノ採譜評価指標。

| 指標 | キー | 内容 |
|------|------|------|
| Onset F1 | `onset_f`, `onset_p`, `onset_r` | ノートの開始時刻のみで判定（pitch 無視） |
| Note F1 | `note_f`, `note_p`, `note_r` | Onset + Offset + Pitch の一致 |
| Note+Vel F1 | `note_vel_f`, `note_vel_p`, `note_vel_r` | 上記 + Velocity の一致 |

**マッチング基準のデフォルト:**
- `onset_tolerance`: 50ms（onset の許容誤差）
- `offset_ratio`: 0.2（offset は duration の 20% 以内、最低 50ms）
- `velocity_tolerance`: 0.1（velocity を [0,1] に正規化した上での許容幅）

### 3.2 コアモジュール `my_mt3/eval.py`

```python
from my_mt3.eval import evaluate_midi_pair, evaluate_directory

# 単一ペア評価
metrics = evaluate_midi_pair("ref.mid", "est.mid", onset_tolerance=0.05, program=0)
# -> {"onset_f": 0.85, "onset_p": 0.80, "onset_r": 0.90, "note_f": ..., ...}

# ディレクトリ一括評価
per_file, summary = evaluate_directory(
    [("ref1.mid", "est1.mid"), ("ref2.mid", "est2.mid")],
    program=0,
)
```

**関数一覧:**

| 関数 | 入力 | 出力 |
|------|------|------|
| `midi_to_intervals_pitches(midi_path, ...)` | MIDI パス | `(intervals [N,2], pitches [N], velocities [N])` |
| `evaluate_midi_pair(ref, est, ...)` | ref/est MIDI パス | 9指標の dict |
| `evaluate_directory(pairs, ...)` | `(ref, est)` ペアのリスト | `(per_file list, summary dict)` |

### 3.3 推論と同時に評価 (`--eval`)

`run/infer_maestro.py --eval` を使うと、データセットモードで推論直後に各ファイルを評価する。

```
推論ループ:
  for each (audio, ref_midi, program_id):
    pred_midi = infer_one_song_piano(...)
    pred_midi.write(out_mid)
    if --eval:
      metrics = evaluate_midi_pair(ref_midi, out_mid, ...)
      → eval_rows に蓄積

終了後:
  → eval_metrics.csv に保存
  → サマリ（平均 onset_f, note_f, note_vel_f 等）を標準出力に表示
```

出力 CSV の各行は 1 ファイルに対応し、以下のカラムを持つ:
```
stem, onset_f, onset_p, onset_r, note_f, note_p, note_r, note_vel_f, note_vel_p, note_vel_r
```

### 3.4 単体評価スクリプト `run/eval_maestro.py`

推論済み MIDI を事後に評価するスタンドアロンスクリプト。

```bash
uv run run/eval_maestro.py \
  --pred_dir outputs/maestro_validation_pred \
  --root maestro-v3.0.0 \
  --split validation \
  --out_csv results/eval.csv
```

- `pred_dir` 内の `*.pred.mid` ファイルを探索
- stem 名（拡張子と `.pred` を除いた部分）で MAESTRO CSV の ref MIDI とマッチング
- マッチした全ペアを評価し、CSV + サマリを出力

### 3.5 評価 CLI オプション (`run/eval_maestro.py`)

| オプション | デフォルト | 説明 |
|------------|-----------|------|
| `--pred_dir` | (必須) | `.pred.mid` ファイルのディレクトリ |
| `--root` | (必須) | MAESTRO v3.0.0 ルート |
| `--split` | `validation` | 対象スプリット |
| `--program` | `0` | 評価対象の MIDI プログラム番号 |
| `--onset_tolerance` | `0.05` | onset 許容誤差 (秒) |
| `--offset_ratio` | `0.2` | offset ratio |
| `--offset_min_tolerance` | `0.05` | 最小 offset 許容誤差 (秒) |
| `--velocity_tolerance` | `0.1` | velocity 許容幅 ([0,1] スケール) |
| `--out_csv` | `pred_dir/eval_metrics.csv` | 出力 CSV パス |

---

## 4. ファイル構成

```
my_mt3/
  infer.py          # 汎用推論関数（greedy_decode, to_midi_from_tokens 等）
  decode_kv.py       # FastDecoderKV / KVCache（KV Cache 実装）
  eval.py            # 評価コア（evaluate_midi_pair, evaluate_directory）
  model.py           # MT3Mini（Encoder + Decoder）

run/
  infer_maestro.py   # MAESTRO Piano 推論（+ --eval オプション）
  eval_maestro.py    # 単体評価スクリプト
  infer_minimal.py   # 汎用最小推論スクリプト（KV Cache 対応済み）
  eval_minimal.py    # 評価プロトタイプ（旧版、参考用）
```

---

## 5. 典型的なワークフロー

### A) 学習後の validation 評価（推論＋評価を一括）

```bash
CUDA_VISIBLE_DEVICES=0 uv run run/infer_maestro.py \
  --ckpt checkpoints_maestro/run_xxx/model_ep500.pt \
  --root maestro-v3.0.0 \
  --split validation \
  --out_dir outputs/val_ep500 \
  --batch_size 32 \
  --eval
```

### B) 事後評価のみ（推論済み MIDI がある場合）

```bash
uv run run/eval_maestro.py \
  --pred_dir outputs/val_ep500 \
  --root maestro-v3.0.0 \
  --split validation
```

### C) 複数 checkpoint の比較

```bash
for ep in 100 500 1000 1800; do
  uv run run/infer_maestro.py \
    --ckpt checkpoints/model_ep${ep}.pt \
    --root maestro-v3.0.0 \
    --split validation \
    --out_dir outputs/val_ep${ep} \
    --eval
done
# → 各 outputs/val_ep*/eval_metrics.csv を比較
```

### D) test split での最終評価

```bash
uv run run/infer_maestro.py \
  --ckpt checkpoints/best_model.pt \
  --root maestro-v3.0.0 \
  --split test \
  --out_dir outputs/test_final \
  --eval
```

---

以上。必要に応じて本資料を更新してください。
