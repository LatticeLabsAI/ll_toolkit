"""ll_gen STEPVAE in native MLX — faithful weight-conversion port.

The ll_gen neural generator is ``ll_stepnet.stepnet.vae.STEPVAE`` (a transformer
encoder-decoder VAE over CAD command-token sequences), trained to the checkpoints
``ll_gen/checkpoints/vae_warm.pt`` / ``vae_rl_solid.pt``. This reproduces that EXACT
architecture in MLX and CONVERTS the real trained weights, so the MLX VAE *is* the
trained model — same weights, same outputs — running natively on Apple Silicon.

Architecture (from the checkpoint + vae.py):
  encoder : STEPTransformerEncoder (token_emb 50000x256, pos 5000x256, 6 post-norm
            TransformerEncoderLayers, final LayerNorm) -> masked-mean pool
  mu_head / log_var_head : Linear(256->256)   (log_var clamped to [-30, 20])
  decode(z):
    z_proj = latent_project(z)                      # Linear 256->256
    hidden = z_proj.broadcast[B,S,256] + dec_pos_embedding[1,60,256]
    z_memory = z_proj.broadcast[B,S,256]
    hidden = decoder._transformer(hidden, memory=z_memory, tgt_mask=causal)  # 6x
              TransformerDecoderLayer: causal self_attn + cross multihead_attn + FFN,
              post-norm with 3 LayerNorms
    hidden = decoder.layer_norm(hidden)
  command_head : Linear(256->6)        param_heads : 16 x Linear(256->256)

Notes:
  * decode is PARALLEL — the decoder input is z broadcast over positions, NOT shifted
    tokens, so decoder.token_embedding / pos_embedding are UNUSED on this path (skipped).
  * eval reparam returns mu (deterministic), so encode->decode(mu) is reproducible.
  * MHA is implemented manually and splits the packed in_proj_weight into Wq/Wk/Wv so
    the same module serves self-attention AND cross-attention (q=tgt, k=v=memory).

Modes: probe | convert | parity.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import logging  # noqa: E402

logging.disable(logging.WARNING)
import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.nn as mlxnn  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]


# --- MLX modules --------------------------------------------------------------
class MHA(mlxnn.Module):
    """nn.MultiheadAttention from a packed in_proj [3d,d]; splits Wq/Wk/Wv so it
    serves both self-attention and cross-attention. Optional additive mask."""

    def __init__(self, d, heads):
        super().__init__()
        self.d, self.h, self.hd = d, heads, d // heads
        self.in_proj = mlxnn.Linear(d, 3 * d)
        self.out_proj = mlxnn.Linear(d, d)

    def __call__(self, query, key, mask=None):
        d = self.d
        w, bvec = self.in_proj.weight, self.in_proj.bias
        wq, wk, wv = w[:d], w[d:2 * d], w[2 * d:]
        bq, bk, bv = bvec[:d], bvec[d:2 * d], bvec[2 * d:]
        q = query @ wq.T + bq
        k = key @ wk.T + bk
        v = key @ wv.T + bv
        b, s, _ = q.shape
        sk = k.shape[1]

        def split(t, n):
            return mx.transpose(t.reshape(b, n, self.h, self.hd), (0, 2, 1, 3))

        q, k, v = split(q, s), split(k, sk), split(v, sk)
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) / (self.hd ** 0.5)  # [b,h,s,sk]
        if mask is not None:
            scores = scores + mask
        ctx = mx.softmax(scores, axis=-1) @ v
        ctx = mx.transpose(ctx, (0, 2, 1, 3)).reshape(b, s, d)
        return self.out_proj(ctx)


class EncLayer(mlxnn.Module):
    """nn.TransformerEncoderLayer, post-norm, relu."""

    def __init__(self, d=256, heads=8, ff=1024):
        super().__init__()
        self.self_attn = MHA(d, heads)
        self.linear1 = mlxnn.Linear(d, ff)
        self.linear2 = mlxnn.Linear(ff, d)
        self.norm1 = mlxnn.LayerNorm(d)
        self.norm2 = mlxnn.LayerNorm(d)

    def __call__(self, x):
        x = self.norm1(x + self.self_attn(x, x))
        return self.norm2(x + self.linear2(mlxnn.relu(self.linear1(x))))


class EncoderMLX(mlxnn.Module):
    def __init__(self, vocab=50000, d=256, layers=6, heads=8, ff=1024):
        super().__init__()
        self.token_embedding = mlxnn.Embedding(vocab, d)
        self.pos_embedding = mx.zeros((1, 5000, d))
        self.layers = [EncLayer(d, heads, ff) for _ in range(layers)]
        self.layer_norm = mlxnn.LayerNorm(d)

    def __call__(self, ids):
        x = self.token_embedding(ids) + self.pos_embedding[:, : ids.shape[1], :]
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class DecLayer(mlxnn.Module):
    """nn.TransformerDecoderLayer, post-norm, relu: causal self-attn + cross-attn + FFN."""

    def __init__(self, d=256, heads=8, ff=1024):
        super().__init__()
        self.self_attn = MHA(d, heads)
        self.multihead_attn = MHA(d, heads)
        self.linear1 = mlxnn.Linear(d, ff)
        self.linear2 = mlxnn.Linear(ff, d)
        self.norm1 = mlxnn.LayerNorm(d)
        self.norm2 = mlxnn.LayerNorm(d)
        self.norm3 = mlxnn.LayerNorm(d)

    def __call__(self, tgt, memory, causal):
        tgt = self.norm1(tgt + self.self_attn(tgt, tgt, mask=causal))
        tgt = self.norm2(tgt + self.multihead_attn(tgt, memory))
        return self.norm3(tgt + self.linear2(mlxnn.relu(self.linear1(tgt))))


class STEPVAEMLX(mlxnn.Module):
    def __init__(self, latent=256, d=256, layers=6, heads=8, ff=1024,
                 max_seq=60, n_cmd=6, n_param=16, n_levels=256):
        super().__init__()
        self.encoder = EncoderMLX(50000, d, layers, heads, ff)
        self.mu_head = mlxnn.Linear(d, latent)
        self.log_var_head = mlxnn.Linear(d, latent)
        self.latent_project = mlxnn.Linear(latent, d)
        self.dec_pos_embedding = mx.zeros((1, max_seq, d))
        self.dec_layers = [DecLayer(d, heads, ff) for _ in range(layers)]
        self.dec_layer_norm = mlxnn.LayerNorm(d)
        self.command_head = mlxnn.Linear(d, n_cmd)
        self.param_heads = [mlxnn.Linear(d, n_levels) for _ in range(n_param)]
        self.d = d

    def encode(self, ids):
        hidden = self.encoder(ids)
        pooled = hidden.mean(axis=1)  # no padding mask -> plain mean
        mu = self.mu_head(pooled)
        log_var = mx.clip(self.log_var_head(pooled), -30, 20)
        return mu, log_var

    def decode(self, z, seq_len):
        b = z.shape[0]
        z_proj = self.latent_project(z)                       # [B,d]
        z_exp = mx.broadcast_to(z_proj[:, None, :], (b, seq_len, self.d))
        hidden = z_exp + self.dec_pos_embedding[:, :seq_len, :]
        # causal mask [S,S]: 0 on/below diag, -inf above
        causal = mx.where(mx.triu(mx.ones((seq_len, seq_len)), k=1) > 0,
                          mx.array(-1e9, mx.float32), mx.array(0.0, mx.float32))
        for layer in self.dec_layers:
            hidden = layer(hidden, z_exp, causal)
        hidden = self.dec_layer_norm(hidden)
        cmd = self.command_head(hidden)                       # [B,S,6]
        params = mx.stack([h(hidden) for h in self.param_heads], axis=0)  # [16,B,S,256]
        return cmd, params


# --- weight conversion --------------------------------------------------------
def convert_checkpoint(ckpt_path, model):
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck if hasattr(ck, "items") else ck.get("model_state_dict", ck)

    def t(key):
        return mx.array(sd[key].detach().cpu().float().numpy())

    pairs = [("encoder.token_embedding.weight", t("encoder.token_embedding.weight")),
             ("encoder.pos_embedding", t("encoder.pos_embedding")),
             ("encoder.layer_norm.weight", t("encoder.layer_norm.weight")),
             ("encoder.layer_norm.bias", t("encoder.layer_norm.bias"))]
    for i in range(6):
        s = f"encoder.transformer.layers.{i}"
        d = f"encoder.layers.{i}"
        pairs += [(f"{d}.self_attn.in_proj.weight", t(f"{s}.self_attn.in_proj_weight")),
                  (f"{d}.self_attn.in_proj.bias", t(f"{s}.self_attn.in_proj_bias")),
                  (f"{d}.self_attn.out_proj.weight", t(f"{s}.self_attn.out_proj.weight")),
                  (f"{d}.self_attn.out_proj.bias", t(f"{s}.self_attn.out_proj.bias")),
                  (f"{d}.linear1.weight", t(f"{s}.linear1.weight")),
                  (f"{d}.linear1.bias", t(f"{s}.linear1.bias")),
                  (f"{d}.linear2.weight", t(f"{s}.linear2.weight")),
                  (f"{d}.linear2.bias", t(f"{s}.linear2.bias")),
                  (f"{d}.norm1.weight", t(f"{s}.norm1.weight")), (f"{d}.norm1.bias", t(f"{s}.norm1.bias")),
                  (f"{d}.norm2.weight", t(f"{s}.norm2.weight")), (f"{d}.norm2.bias", t(f"{s}.norm2.bias"))]
    for src, dst in (("mu_head", "mu_head"), ("log_var_head", "log_var_head"),
                     ("latent_project", "latent_project"), ("command_head", "command_head")):
        pairs += [(f"{dst}.weight", t(f"{src}.weight")), (f"{dst}.bias", t(f"{src}.bias"))]
    pairs.append(("dec_pos_embedding", t("dec_pos_embedding")))
    pairs += [("dec_layer_norm.weight", t("decoder.layer_norm.weight")),
              ("dec_layer_norm.bias", t("decoder.layer_norm.bias"))]
    for i in range(6):
        s = f"decoder._transformer.layers.{i}"
        d = f"dec_layers.{i}"
        for a, mlxa in (("self_attn", "self_attn"), ("multihead_attn", "multihead_attn")):
            pairs += [(f"{d}.{mlxa}.in_proj.weight", t(f"{s}.{a}.in_proj_weight")),
                      (f"{d}.{mlxa}.in_proj.bias", t(f"{s}.{a}.in_proj_bias")),
                      (f"{d}.{mlxa}.out_proj.weight", t(f"{s}.{a}.out_proj.weight")),
                      (f"{d}.{mlxa}.out_proj.bias", t(f"{s}.{a}.out_proj.bias"))]
        pairs += [(f"{d}.linear1.weight", t(f"{s}.linear1.weight")), (f"{d}.linear1.bias", t(f"{s}.linear1.bias")),
                  (f"{d}.linear2.weight", t(f"{s}.linear2.weight")), (f"{d}.linear2.bias", t(f"{s}.linear2.bias")),
                  (f"{d}.norm1.weight", t(f"{s}.norm1.weight")), (f"{d}.norm1.bias", t(f"{s}.norm1.bias")),
                  (f"{d}.norm2.weight", t(f"{s}.norm2.weight")), (f"{d}.norm2.bias", t(f"{s}.norm2.bias")),
                  (f"{d}.norm3.weight", t(f"{s}.norm3.weight")), (f"{d}.norm3.bias", t(f"{s}.norm3.bias"))]
    n_param = sum(1 for k in sd if k.startswith("param_heads.") and k.endswith(".weight"))
    for i in range(n_param):
        pairs += [(f"param_heads.{i}.weight", t(f"param_heads.{i}.weight")),
                  (f"param_heads.{i}.bias", t(f"param_heads.{i}.bias"))]
    model.update(tree_unflatten(pairs))
    mx.eval(model.parameters())
    return len(pairs)


def parity(ckpt):
    sys.path.insert(0, str(_REPO / "ll_stepnet"))
    import types

    import torch
    from stepnet.vae import STEPVAE

    cfg = types.SimpleNamespace(token_embed_dim=256, vocab_size=50000,
                                num_transformer_layers=6, dropout=0.1)
    pt = STEPVAE(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck if hasattr(ck, "items") else ck.get("model_state_dict", ck)
    pt.load_state_dict(sd, strict=False)
    pt.eval()
    S = int(pt.max_seq_len)
    n_param = len(pt.param_heads)

    model = STEPVAEMLX(max_seq=S, n_param=n_param)
    npar = convert_checkpoint(ckpt, model)

    rng = np.random.default_rng(0)
    ids = rng.integers(1, 268, (3, S)).astype(np.int64)

    with torch.no_grad():
        mu_pt, lv_pt = pt.encode(torch.from_numpy(ids))
        hidden_pt = pt.decode(mu_pt, seq_len=S)
        cmd_pt = pt.command_head(hidden_pt).numpy()
        par_pt = np.stack([h(hidden_pt).numpy() for h in pt.param_heads])  # [P,B,S,256]

    mu_mlx, _ = model.encode(mx.array(ids))
    cmd_mlx, par_mlx = model.decode(mu_mlx, S)
    cmd_mlx = np.array(cmd_mlx.tolist()); par_mlx = np.array(par_mlx.tolist())

    d_mu = float(np.abs(np.array(mu_mlx.tolist()) - mu_pt.numpy()).max())
    d_cmd = float(np.abs(cmd_mlx - cmd_pt).max())
    d_par = float(np.abs(par_mlx - par_pt).max())
    # decoded-command agreement (argmax over the 6 command types, per position)
    agree = float((cmd_mlx.argmax(-1) == cmd_pt.argmax(-1)).mean())
    print(f"converted {npar} tensors from {os.path.basename(ckpt)}", flush=True)
    print(f"PARITY max-abs-diff  mu={d_mu:.2e}  command_logits={d_cmd:.2e}  param_logits={d_par:.2e}", flush=True)
    print(f"decoded-command argmax agreement = {agree:.4f}", flush=True)
    ok = d_mu < 1e-3 and d_cmd < 1e-3 and d_par < 1e-3 and agree > 0.999
    out = str(_REPO / "ll_gen/checkpoints/vae_mlx.safetensors")
    mx.save_safetensors(out, dict(tree_flatten(model.parameters())))
    print(f"FAITHFUL_PARITY {'PASS' if ok else 'FAIL'}  -> saved {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "convert", "parity"], default="parity")
    ap.add_argument("--ckpt", default=str(_REPO / "ll_gen/checkpoints/vae_warm.pt"))
    args = ap.parse_args()
    if args.mode == "probe":
        m = STEPVAEMLX()
        mu, lv = m.encode(mx.array(np.random.randint(1, 268, (2, 60))))
        cmd, par = m.decode(mu, 60)
        print(f"probe: mu {mu.shape} command {cmd.shape} params {par.shape} "
              f"finite={bool(mx.isfinite(cmd).all().item())}", flush=True)
        return
    if args.mode == "convert":
        m = STEPVAEMLX()
        n = convert_checkpoint(args.ckpt, m)
        mx.save_safetensors(str(_REPO / "ll_gen/checkpoints/vae_mlx.safetensors"),
                            dict(tree_flatten(m.parameters())))
        print(f"converted {n} tensors", flush=True)
        return
    parity(args.ckpt)


if __name__ == "__main__":
    main()
