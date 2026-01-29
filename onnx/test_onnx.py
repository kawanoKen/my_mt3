import numpy as np
import onnxruntime as ort

enc_sess = ort.InferenceSession("encoder.onnx", providers=["CPUExecutionProvider"])
dec_sess = ort.InferenceSession("decoder_last.onnx", providers=["CPUExecutionProvider"])

# 例: mel [1,T,M]
mel = np.random.randn(1, 256, 256).astype(np.float32)
mem = enc_sess.run(None, {"mel": mel})[0]

# 初期トークン（あなたの PRG_0 相当）
bos = 123  # ←あなたの VOCAB から実値を入れる
# decoder.onnx はエクスポート時の L（既定=4）に合わせた形状が埋め込まれているため、同じ長さで入力
y = np.array([[bos, bos, bos, bos]], dtype=np.int64)

logits = dec_sess.run(None, {"y": y, "mem": mem})[0]  # [1,V]
nxt = int(logits.argmax(axis=-1)[0])
print("next token:", nxt)
