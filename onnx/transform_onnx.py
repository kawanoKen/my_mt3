import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from my_mt3.model import MT3Mini
from my_mt3.infer import greedy_decode, to_midi_from_tokens

from pathlib import Path
from my_mt3.tokenizer import VOCAB
import argparse




class EncWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, mel):
        return self.model.enc(mel)

class DecLastWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, y, mem):
        logits = self.model.dec(y, mem)   # [B, L, V]想定
        return logits[:, -1, :]           # [B, V]

def build_model(ckpt_path: Path, device: str):
    model = MT3Mini(vocab_size=len(VOCAB.itos))
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model

def export_onnx(model, T=256, M=None, L=4, opset=17):
    device = "cpu"
    model = model.to(device).eval()

    enc = EncWrapper(model).eval()
    dec = DecLastWrapper(model).eval()

    # モデルの想定メル次元に合わせる（未指定なら自動検出）
    if M is None:
        if hasattr(model.enc, "proj") and hasattr(model.enc.proj, "in_features"):
            M = int(model.enc.proj.in_features)
        else:
            M = 256  # フォールバック

    dummy_mel = torch.randn(1, T, M, dtype=torch.float32, device=device)
    dummy_y   = torch.ones(1, L, dtype=torch.long, device=device)

    # encoder
    torch.onnx.export(
        enc, (dummy_mel,), "encoder.onnx",
        opset_version=opset,
        input_names=["mel"], output_names=["mem"],
        dynamic_axes={"mel": {1: "T"}},  # まずTだけ可変（最短）
    )

    # memを作ってdecoder export
    with torch.no_grad():
        dummy_mem = enc(dummy_mel)

    torch.onnx.export(
        dec, (dummy_y, dummy_mem), "decoder_last.onnx",
        opset_version=opset,
        input_names=["y", "mem"], output_names=["logits_last"],
        dynamic_axes={"y": {1: "L"}},  # y長だけ可変（最短）
    )

# 使い方
# export_onnx(model, T=?, M=?, L=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    args = parser.parse_args()
    model = build_model(args.ckpt, device="cpu")
    export_onnx(model)

if __name__ == "__main__":
    main()