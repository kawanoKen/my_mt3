# my_mt3: Music Transcription with Transformers

MT3（Music Transcription with Transformers）のPython実装です。音声ファイルからMIDI情報への自動音楽転写を行います。

## 🎵 概要

このプロジェクトは、深層学習を用いて音声信号から音楽の音符情報を自動抽出するシステムです。Transformerアーキテクチャを基盤とし、音声をlog-Melスペクトログラムに変換してから、楽器・音程・タイミング情報をトークン列として予測します。

### 主な特徴

- **torchaudio基盤**: PyTorchエコシステムとの統合、GPU加速対応
- **時系列トークン化**: 10ms精度での高精細な音楽イベント表現
- **マルチ楽器対応**: piano, guitar, bass, drums, vocal（MVP版）
- **チャンク処理**: 長時間音声の効率的な分割処理
- **エンドツーエンド**: 音声ファイルからMIDIまでの完全な処理パイプライン

## 📁 プロジェクト構造

```
my_mt3/
├── my_mt3/                 # コアライブラリ
│   ├── audio.py           # 音声処理（読み込み、Melスペクトログラム変換）
│   ├── tokenizer.py       # 音楽イベントのトークン化
│   ├── dataset.py         # PyTorchデータセット
│   ├── model.py           # Transformerモデル定義
│   ├── train.py           # 訓練ループ
│   ├── infer.py           # 推論処理
│   ├── metrics.py         # 評価指標
│   └── utils.py           # ユーティリティ関数
├── run/                   # 実行スクリプト
│   ├── train_minimal.py   # 最小限の訓練例
│   └── make_synth_piano.py # 合成データ生成
├── data/                  # データディレクトリ
│   ├── wavs/             # 音声ファイル
│   └── midis/            # MIDIファイル
└── main.py               # メインエントリーポイント
```

## 🚀 セットアップ

### 必要環境

- Python 3.10以上
- PyTorch 2.8以上
- torchaudio 2.8以上

### インストール

```bash
# リポジトリのクローン
git clone https://github.com/kawanoKen/my_mt3.git
cd my_mt3

# 依存関係のインストール（uvを使用）
uv sync

# または pip を使用
pip install -e .
```

### 依存ライブラリ

- `torch` / `torchaudio`: 深層学習フレームワーク
- `numpy`: 数値計算
- `pretty-midi`: MIDI処理
- `soundfile`: 音声ファイル処理

## 💡 使用方法

### 1. 合成データの生成

```bash
python run/make_synth_piano.py
```

### 2. モデルの訓練

```bash
python run/train_minimal.py
```

### 3. 推論の実行

```python
from my_mt3 import load_wav_mono, wav_to_logmel, MT3Model, encode_events

# 音声読み込み
audio, sr = load_wav_mono("path/to/audio.wav")

# Melスペクトログラム変換
logmel = wav_to_logmel(audio, sr)

# モデル推論
model = MT3Model()
tokens = model.predict(logmel)

# 結果の後処理
events = decode_events(tokens)  # 実装予定
```

## 🎼 音楽表現形式

### トークン体系

音楽イベントは以下のトークンで表現されます：

- `PRG_x`: 楽器プログラム（0=piano, 1=guitar, ...）
- `TIM_x`: タイムステップ（10ms刻み、0-204）
- `NON_x`: ノートオン（音符開始、0-127）
- `NOF_x`: ノートオフ（音符終了、0-127）
- `<eos>`: シーケンス終了
- `<end_tie>`: 音の継続終了

### エンコード例

```
[PRG_0, TIM_0, NON_60, TIM_50, NOF_60, NON_64, TIM_100, NOF_64, <eos>]
```

これは「ピアノで、0ms時点でC4開始、50ms時点でC4終了・E4開始、100ms時点でE4終了」を表現

## 🔧 技術仕様

### 音声処理

- **サンプリングレート**: 22.05kHz
- **FFTサイズ**: 2048
- **ホップ長**: 256（約11.6ms）
- **Melバンド数**: 256
- **チャンク長**: 2.048秒

### モデルアーキテクチャ

- **ベース**: Transformer（Encoder-Decoder）
- **入力**: log-Melスペクトログラム [時間, 256]
- **出力**: 音楽イベントトークン列
- **語彙サイズ**: 約770トークン

## 📊 評価指標

- **フレーム精度**: 時間軸での音符検出精度
- **音符F1スコア**: 音符レベルでの検出性能
- **楽器分類精度**: 楽器種別の判定精度

## 🛠️ 開発状況

### 実装済み

- ✅ 音声処理（torchaudio版）
- ✅ 音楽イベントトークン化
- ✅ データセット構築
- ✅ 基本モデル定義
- ✅ 訓練パイプライン
- ✅ 合成データ生成

### 今後の実装予定

- ⏳ デコード関数（トークン→MIDI変換）
- ⏳ より高度な評価指標
- ⏳ 複数楽器同時転写
- ⏳ リアルタイム処理
- ⏳ Web UI

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 コントリビューション

プルリクエストやイシューの報告を歓迎いたします。

## 📞 お問い合わせ

プロジェクトに関するご質問は、GitHubのIssuesページでお願いいたします。

---

**Note**: このプロジェクトはMVP（Minimum Viable Product）として設計されており、研究・教育目的での使用を想定しています。
