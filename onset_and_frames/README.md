# Onsets and Frames + Semi-Supervised Pseudo-Labeling (PyTorch)

このリポジトリは、以下 2 本の論文の内容に沿って **PyTorch ベースで再現実装**するための最小コード一式です。

- Hawthorne et al., **"Onsets and Frames: Dual-Objective Piano Transcription"** (ISMIR 2018)
- Strahl & Müller, **"Semi-Supervised Piano Transcription Using Pseudo-Labeling Techniques"** (ISMIR 2024)

本実装は「研究再現/学習用」を目的としたもので、学習をそのまま走らせるには
MAPS / MAESTRO などのデータセットの入手と前処理が必要です（論文に従う）。

---

## 1. セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> 注意: `pretty_midi` と `mir_eval` は評価や MIDI 入出力に使います（任意ですが推奨）。

---

## 2. データの用意（マニフェスト）

### 2.1 ラベル付きデータ (audio + midi)
CSV を用意します（ヘッダ必須）。

`manifests/labeled_train.csv`
```csv
audio_path,midi_path
/path/to/file1.wav,/path/to/file1.mid
/path/to/file2.wav,/path/to/file2.mid
...
```

### 2.2 ラベルなしデータ (audioのみ)
`manifests/unlabeled_train.csv`
```csv
audio_path
/path/to/unlabeled1.wav
/path/to/unlabeled2.wav
...
```

---

## 3. 学習

### 3.1 Supervised (O&F)
```bash
python -m oaf.train_supervised \
  --train_csv manifests/labeled_train.csv \
  --valid_csv manifests/labeled_valid.csv \
  --out_dir runs/of_supervised
```

### 3.2 Semi-supervised (OF-SS4)
論文(ISMIR2024)に合わせて、まず supervised で 50k iter 学習 → その重みを初期値に SSL を追加して 50k iter 学習、という流れを想定しています。

```bash
# まず supervised を回して checkpoint を作る（例）
python -m oaf.train_supervised --train_csv ... --out_dir runs/of_pretrain

# つぎに SSL（OF-SS4）
python -m oaf.train_ssl \
  --labeled_csv manifests/labeled_train.csv \
  --unlabeled_csv manifests/unlabeled_train.csv \
  --valid_csv manifests/labeled_valid.csv \
  --init_ckpt runs/of_pretrain/checkpoints/last.pt \
  --out_dir runs/of_ss4
```

---

## 4. 推論 / デコード

```bash
python -m oaf.transcribe \
  --ckpt runs/of_ss4/checkpoints/best.pt \
  --audio /path/to/test.wav \
  --out_midi out.mid
```

---

## 5. 実装メモ（どこに何があるか）

- `oaf/model.py` : Onsets and Frames (onset/frame/offset/velocity) のモデル
- `oaf/features.py` : 16kHz, n_fft=2048, hop=512, n_mels=229 の log-mel 特徴
- `oaf/labels.py` : MIDI から onset/frame/offset のフレームラベル作成（+ sustain 処理）
- `oaf/decoding.py` : onset で frame をゲートしてノートイベントに復元（論文通り）
- `oaf/ssl.py` : pseudo-labeling, consistency regularization, distribution matching
- `oaf/augment.py` : 周波数マスク + ガウスノイズ（論文の強い augmentation）
- `oaf/losses.py` : supervised / unsupervised loss（NaN=ignore を mask として実装）
- `oaf/tune_thresholds.py` : τ_on, τ_fr を validation でチューニング（任意）

---

## 6. 免責

- 論文実験の「完全一致」には、元実装の細部（正確な conv-stack 構成、正規化、データ分割、MIDI の扱い、しきい値探索の範囲など）も合わせる必要があります。
- ただし本コードは **論文で明記されている枠組み**（onset+frame のゲーティング、pseudo-labeling + consistency + distribution matching、損失の構造と重み付け、augmentation）を忠実に実装しています。
