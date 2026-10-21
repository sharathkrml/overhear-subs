"""Unpack Bark's `.bark` weights into an mlx-audio Parakeet folder.

Bark 3.0 ships its Parakeet-TDT fine-tune as one entropy-coded blob rather than
safetensors: per-row level tables plus ANS-coded codes for the encoder linears,
ANS-coded integers for the other matrices, fp16 for the small tensors, and LZMA
on the side blobs. Nothing here is standard, so the decoder is vendored from the
model release (`bark_loader.py` / `bark_mlx.py`, CC-BY-4.0, Conway Research)
and pinned by revision in `backends.BARK_REVISION`.

Only numpy is needed to decode; mlx only writes the unpacked folder.
"""

from __future__ import annotations

import json
import lzma
import re
import shutil
from pathlib import Path

import numpy as np

NC = 16  # column classes per level-coded tensor

# transformers ParakeetForTDT names/layouts -> mlx-audio Parakeet names/layouts
# (channels-last convs, fused LSTM bias).
_RENAMES = [
    (r"encoder\.subsampling\.layers\.", "encoder.pre_encode.conv."),
    (r"encoder\.subsampling\.linear\.", "encoder.pre_encode.out."),
    (r"\.conv\.norm\.", ".conv.batch_norm."),
    (r"\.q_proj\.", ".linear_q."), (r"\.k_proj\.", ".linear_k."),
    (r"\.v_proj\.", ".linear_v."), (r"\.o_proj\.", ".linear_out."),
    (r"\.relative_k_proj\.", ".linear_pos."),
    (r"\.bias_u$", ".pos_bias_u"), (r"\.bias_v$", ".pos_bias_v"),
    (r"^decoder\.embedding\.", "decoder.prediction.embed."),
    (r"^encoder_projector\.", "joint.enc."),
    (r"^decoder\.decoder_projector\.", "joint.pred."),
    (r"^joint\.head\.", "joint.joint_net.2."),
]

_FILES = ("config.json", "tokenizer.model", "tokenizer.vocab", "vocab.txt")


def _decode_codes(cls, counts, words, rows, cols):
    import constriction
    dec = constriction.stream.stack.AnsCoder(words)
    codes = np.empty((rows, cols), np.uint8)
    for c in range(len(counts)):
        idx = np.nonzero(cls == c)[0]
        model = constriction.stream.model.Categorical(
            counts[c] / counts[c].sum(), perfect=False)
        codes[:, idx] = dec.decode(model, rows * len(idx)).reshape(len(idx), rows).T
    return codes


def _levels_to_weights(lev, codes):
    K = lev.shape[1]
    lev = lev.astype(np.float32)
    mag = np.where(codes == 0, 0, np.where(codes <= K, codes, codes - K)).astype(np.int64)
    val = np.take_along_axis(
        np.concatenate([np.zeros((lev.shape[0], 1), np.float32), lev], 1), mag, 1)
    return np.where(codes > K, -val, val)


def _unz(b, flag):
    return lzma.decompress(b) if flag else b


def load_state_dict(path: Path) -> dict[str, np.ndarray]:
    """`.bark` -> {transformers weight name: numpy array}."""
    import constriction
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[:8], "little")
    head = json.loads(raw[8:8 + n])
    if head["format"] not in ("bark-levels-v2", "bark-levels-v3"):
        raise ValueError(f"unknown bark format {head['format']!r}")
    p = 8 + n
    sd: dict[str, np.ndarray] = {}
    for e in head["index"]:
        shape = e["shape"]
        if e["k"] == "levels2":
            K, rows, cols = e["K"], shape[0], int(np.prod(shape[1:]))
            lev = np.frombuffer(_unz(raw[p:p + e["lev"]], e.get("zl")),
                                np.float16).reshape(rows, K)
            p += e["lev"]
            meta = raw[p:p + e["meta"]]
            p += e["meta"]
            nb = (cols * 4 + 7) // 8
            bits = np.unpackbits(np.frombuffer(meta[:nb], np.uint8))[:cols * 4].reshape(cols, 4)
            cls = np.packbits(
                np.concatenate([np.zeros((cols, 4), np.uint8), bits], 1), axis=1)[:, 0]
            counts = np.frombuffer(
                meta[nb:], np.uint32).reshape(NC, 2 * K + 1).astype(np.float64)
            words = np.frombuffer(raw[p:p + e["b"]], np.uint32).copy()
            p += e["b"]
            codes = _decode_codes(cls, counts, words, rows, cols)
            sd[e["n"]] = _levels_to_weights(lev, codes).reshape(shape)
        elif e["k"].startswith("ansint"):
            bits = int(e["k"][6:])
            scale = np.frombuffer(_unz(raw[p:p + e["s"]], e.get("zs")),
                                  np.float16).astype(np.float32)
            p += e["s"]
            counts = np.frombuffer(raw[p:p + e["meta"]], np.uint32).astype(np.float64)
            p += e["meta"]
            words = np.frombuffer(raw[p:p + e["b"]], np.uint32).copy()
            p += e["b"]
            model = constriction.stream.model.Categorical(
                counts / counts.sum(), perfect=False)
            sym = constriction.stream.stack.AnsCoder(words).decode(
                model, int(np.prod(shape)))
            q = (sym - (2 ** (bits - 1) - 1)).astype(np.float32)
            sd[e["n"]] = (q.reshape(len(scale), -1) * scale[:, None]).reshape(shape)
        elif e["k"] == "raw":
            dtype = {"torch.int64": np.int64,
                     "torch.int32": np.int32}.get(e["dtype"], np.float16)
            sd[e["n"]] = np.frombuffer(_unz(raw[p:p + e["b"]], e.get("zr")),
                                       dtype).copy().reshape(shape)
            p += e["b"]
        else:
            raise ValueError(e["k"])
    return sd


def to_mlx_names(sd: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for k, v in sd.items():
        v = np.asarray(v, np.float32)
        if k.endswith("num_batches_tracked") or k.startswith("vad_head"):
            continue
        m = re.match(r"decoder\.lstm\.(weight_ih|weight_hh|bias_ih|bias_hh)_l(\d)", k)
        if m:
            kind, i = m.groups()
            base = f"decoder.prediction.dec_rnn.lstm.{i}"
            if kind == "weight_ih":
                out[f"{base}.Wx"] = v
            elif kind == "weight_hh":
                out[f"{base}.Wh"] = v
            else:
                out[f"{base}.bias"] = out.get(f"{base}.bias", 0) + v
            continue
        if re.search(r"encoder\.subsampling\.layers\.\d+\.weight$", k):
            v = v.transpose(0, 2, 3, 1)
        if re.search(r"conv\.(depthwise_conv|pointwise_conv[12])\.weight$", k):
            v = v.transpose(0, 2, 1)
        for pat, rep in _RENAMES:
            k = re.sub(pat, rep, k)
        out[k] = v
    return out


def prepare(bark_file: Path, meta_dir: Path, out_dir: Path) -> Path:
    """Write the mlx-audio folder once, fp16 (the evaluated precision)."""
    out = Path(out_dir)
    done = out / "model.safetensors"
    if done.is_file():
        return out

    import mlx.core as mx

    out.mkdir(parents=True, exist_ok=True)
    for name in _FILES:
        shutil.copyfile(Path(meta_dir) / name, out / name)
    weights = {k: mx.array(v).astype(mx.float16)
               for k, v in to_mlx_names(load_state_dict(bark_file)).items()}
    # Write beside the target then rename: a half-written safetensors must not
    # pass the is_file() check on the next launch. mlx appends the extension.
    tmp = out / "model.safetensors.tmp.safetensors"
    mx.save_safetensors(str(tmp), weights)
    tmp.replace(done)
    return out
