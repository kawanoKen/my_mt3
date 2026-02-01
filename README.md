# my_mt3: Music Transcription with Transformers

音声→MIDI 転写のtorch使用最小実装です（16kHz log-Mel + Transformer）。
groove midi datasetのみの使用を想定した実装です。

## セットアップ

```bash
git clone https://github.com/kawanoKen/my_mt3.git
cd my_mt3
uv sync     # または: pip install -e .
```

## 学習（GrooveMIDI）

- `dataset/groove/` に `info.csv` と音声/MIDIを配置（列: split, audio_filename, midi_filename）

```bash
python run/train_minimal.py
```

## 推論

```bash
python run/infer_minimal.py \
  --ckpt checkpoints/run_YYYYmmdd_HHMMSS/model_ep0100.pt \
  --root dataset/groove --split test \
  --out_dir outputs/groove_test_pred
```
