"""Latent diffusion that produces VALID CAD — the real fix for the diffusion path.

The shipped diffusion (ll_stepnet StructuredDiffusion + GeometryCodec) denoises raw
B-rep geometry — independent face UV-grids + edge polylines — then tries to SEW them.
Independently-sampled faces never share exact boundaries, so the sewer cannot close a
solid: honest validity 0.0 (its own docstring documents the dead-end).

The robust fix (DeepCAD's actual generative design) changes the REPRESENTATION: diffuse
in the latent of a CAD-PROGRAM autoencoder, and decode with an execution-respecting
autoregressive decoder. The decoder emits a construction program the OCC kernel builds
into a watertight solid, so validity is high — and it comes from the decoder; the
diffusion supplies the latent prior (controllable, unconditional sampling).

Architecture:
  SeqAutoencoder (deterministic, NOT a VAE — avoids posterior collapse):
    encoder : embed + bidirectional transformer + mean-pool -> Linear -> z [d_z]
    decoder : z-conditioned causal transformer (z added at every position); trained
              teacher-forced with WORD-DROPOUT on the decoder inputs so the latent must
              carry the global program (else the AR decoder ignores z).
  LatentDDPM : a denoiser MLP over the (normalised) z's; standard DDPM eps-prediction.

Honest acceptance bar (set in advance): validity of DIFFUSION-SAMPLED z
(z_T -> denoise -> z_0 -> decode -> execute), through the real OCC kernel gated on a
non-degenerate solid (solid + volume>1e-4), with num_distinct > 1, beating the z=0
predict-the-mean baseline. Reconstruction validity is reported too but is NOT the bar.

Modes: probe | train  (train: AE -> DDPM -> measured sampled-z validity).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as mlxnn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "ll_gen/mlx"))

# reuse the AR generator's validated tokenizer / executor / honest validity gate
from ar_generator_mlx import (  # noqa: E402
    MAX_LEN, VOCAB, PAD, BOS, SEQ_EOS, CMD_TOK,
    build_dataset, decode_tokens, command_dicts, make_evaluator,
)
from collections import Counter  # noqa: E402

MASK_TOK = 3  # unused id in the vocab, used as the word-dropout placeholder


# --- transformer blocks -------------------------------------------------------
class Block(mlxnn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.h, self.hd, self.d = heads, d // heads, d
        self.qkv = mlxnn.Linear(d, 3 * d)
        self.proj = mlxnn.Linear(d, d)
        self.n1 = mlxnn.LayerNorm(d)
        self.n2 = mlxnn.LayerNorm(d)
        self.fc1 = mlxnn.Linear(d, ff)
        self.fc2 = mlxnn.Linear(ff, d)

    def __call__(self, x, causal):
        b, s, _ = x.shape
        h = self.n1(x)
        q, k, v = mx.split(self.qkv(h), 3, axis=-1)

        def sp(t):
            return mx.transpose(t.reshape(b, s, self.h, self.hd), (0, 2, 1, 3))

        q, k, v = sp(q), sp(k), sp(v)
        att = (q @ mx.transpose(k, (0, 1, 3, 2))) / (self.hd ** 0.5)
        if causal is not None:
            att = att + causal
        ctx = mx.softmax(att, axis=-1) @ v
        ctx = mx.transpose(ctx, (0, 2, 1, 3)).reshape(b, s, self.d)
        x = x + self.proj(ctx)
        return x + self.fc2(mlxnn.gelu(self.fc1(self.n2(x))))


class SeqAutoencoder(mlxnn.Module):
    def __init__(self, vocab=VOCAB, d=256, d_z=64, enc_layers=3, dec_layers=4, heads=8, ff=1024):
        super().__init__()
        self.embed = mlxnn.Embedding(vocab, d)
        self.pos = mx.zeros((1, MAX_LEN, d))
        self.enc_blocks = [Block(d, heads, ff) for _ in range(enc_layers)]
        self.enc_norm = mlxnn.LayerNorm(d)
        self.to_z = mlxnn.Linear(d, d_z)
        self.from_z = mlxnn.Linear(d_z, d)
        self.dec_blocks = [Block(d, heads, ff) for _ in range(dec_layers)]
        self.dec_norm = mlxnn.LayerNorm(d)
        self.head = mlxnn.Linear(d, vocab)
        self.d, self.d_z = d, d_z

    def encode(self, ids):
        x = self.embed(ids) + self.pos[:, : ids.shape[1], :]
        for blk in self.enc_blocks:
            x = blk(x, None)            # bidirectional
        x = self.enc_norm(x)
        m = (ids != PAD).astype(x.dtype)[..., None]
        pooled = (x * m).sum(axis=1) / mx.maximum(m.sum(axis=1), 1)
        return self.to_z(pooled)        # [B, d_z]

    def decode(self, ids, z):
        s = ids.shape[1]
        causal = mx.where(mx.triu(mx.ones((s, s)), k=1) > 0,
                          mx.array(-1e9, mx.float32), mx.array(0.0, mx.float32))[None, None]
        zc = self.from_z(z)[:, None, :]  # [B,1,d] broadcast over positions
        x = self.embed(ids) + self.pos[:, :s, :] + zc
        for blk in self.dec_blocks:
            x = blk(x, causal)
        return self.head(self.dec_norm(x))


# --- latent DDPM --------------------------------------------------------------
class TimeEmbed(mlxnn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.fc1 = mlxnn.Linear(dim, dim)
        self.fc2 = mlxnn.Linear(dim, dim)

    def __call__(self, t):  # t: [B] in [0,1]
        half = self.dim // 2
        freqs = mx.exp(-np.log(10000) * mx.arange(half) / half)
        a = t[:, None] * freqs[None]
        emb = mx.concatenate([mx.sin(a), mx.cos(a)], axis=-1)
        return self.fc2(mlxnn.gelu(self.fc1(emb)))


class Denoiser(mlxnn.Module):
    def __init__(self, d_z=64, hidden=512, tdim=128):
        super().__init__()
        self.time = TimeEmbed(tdim)
        self.inp = mlxnn.Linear(d_z + tdim, hidden)
        self.h1 = mlxnn.Linear(hidden, hidden)
        self.h2 = mlxnn.Linear(hidden, hidden)
        self.out = mlxnn.Linear(hidden, d_z)

    def __call__(self, z, t):
        te = self.time(t)
        x = mlxnn.gelu(self.inp(mx.concatenate([z, te], axis=-1)))
        x = x + mlxnn.gelu(self.h1(x))
        x = x + mlxnn.gelu(self.h2(x))
        return self.out(x)


class DDPM:
    """Standard DDPM schedule + sampling for a flat latent vector."""

    def __init__(self, steps=200):
        self.steps = steps
        betas = np.linspace(1e-4, 0.02, steps).astype(np.float32)
        alphas = 1.0 - betas
        abar = np.cumprod(alphas)
        self.betas = mx.array(betas)
        self.alphas = mx.array(alphas)
        self.abar = mx.array(abar)
        self._abar_np = abar

    def q_sample(self, z0, t_idx, noise):
        ab = mx.sqrt(self.abar[t_idx])[:, None]
        omab = mx.sqrt(1 - self.abar[t_idx])[:, None]
        return ab * z0 + omab * noise

    def sample(self, denoiser, n, d_z):
        z = mx.random.normal((n, d_z))
        for i in range(self.steps - 1, -1, -1):
            t = mx.full((n,), i / self.steps)
            eps = denoiser(z, t)
            a = self.alphas[i]
            ab = self.abar[i]
            coef = (1 - a) / mx.sqrt(1 - ab)
            mean = (z - coef * eps) / mx.sqrt(a)
            if i > 0:
                z = mean + mx.sqrt(self.betas[i]) * mx.random.normal((n, d_z))
            else:
                z = mean
            mx.eval(z)
        return z


# --- decoding (autoregressive, conditioned on z) ------------------------------
def ar_decode(ae, z, temperature=1.0, top_k=20):
    n = z.shape[0]
    cur = mx.full((n, 1), BOS, dtype=mx.int32)
    done = np.zeros(n, bool)
    seqs = [[] for _ in range(n)]
    for _ in range(MAX_LEN - 1):
        logits = ae.decode(cur, z)[:, -1, :] / temperature
        if top_k:
            kth = mx.sort(logits, axis=-1)[:, -top_k][:, None]
            logits = mx.where(logits < kth, mx.array(-1e9, mx.float32), logits)
        nxt = mx.random.categorical(logits)
        mx.eval(nxt)
        nn = np.array(nxt.tolist())
        for i in range(n):
            if not done[i]:
                t = int(nn[i])
                seqs[i].append(t)
                if t == SEQ_EOS or t == CMD_TOK["EOS"]:
                    done[i] = True
        cur = mx.concatenate([cur, nxt[:, None].astype(mx.int32)], axis=1)
        if done.all():
            break
    return seqs


def measure(seqs, evaluate):
    valid, vols, sigs = 0, [], []
    for s in seqs:
        ok, vol, sig = evaluate(s)
        if ok:
            valid += 1
            vols.append(vol)
            sigs.append(sig)
    top = (Counter(sigs).most_common(1)[0][1] / len(sigs)) if sigs else 0.0
    return {"validity": valid / len(seqs), "num_valid": valid, "num_distinct": len(set(sigs)),
            "top_shape_frac": round(top, 3), "mean_volume": float(np.mean(vols)) if vols else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "train"], default="train")
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--ae-epochs", type=int, default=40)
    ap.add_argument("--dm-epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--d-z", type=int, default=64)
    ap.add_argument("--word-dropout", type=float, default=0.5)
    ap.add_argument("--ddpm-steps", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--out", default=str(_REPO / "ll_gen/checkpoints"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cache = f"{args.out}/ar_tokens_cache.npz"

    if args.mode == "probe":
        toks = build_dataset(256, cache)
        ae = SeqAutoencoder(d_z=args.d_z)
        z = ae.encode(mx.array(toks[:8]))
        logits = ae.decode(mx.array(toks[:8]), z)
        print(f"probe: tokens {toks.shape} z {z.shape} dec_logits {logits.shape}", flush=True)
        return

    print("loading real DeepCAD command sequences ...", flush=True)
    toks = build_dataset(args.n_train, cache)
    print(f"{toks.shape[0]} sequences", flush=True)
    evaluate = make_evaluator()
    ae = SeqAutoencoder(d_z=args.d_z)
    opt = optim.AdamW(learning_rate=3e-4, weight_decay=0.01)

    def ae_loss(ids):
        # word-dropout on decoder inputs forces the latent to carry global structure
        z = ae.encode(ids)
        din = ids[:, :-1]
        keep = (mx.random.uniform(shape=din.shape) > args.word_dropout)
        din = mx.where(keep, din, mx.array(MASK_TOK, mx.int32))
        logits = ae.decode(din, z)
        tgt = ids[:, 1:]
        m = (tgt != PAD).astype(mx.float32)
        ce = mlxnn.losses.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none")
        ce = ce.reshape(tgt.shape) * m
        zreg = 1e-4 * (z * z).mean()           # keep the latent bounded -> diffusable
        return ce.sum() / mx.maximum(m.sum(), 1) + zreg

    lg = mlxnn.value_and_grad(ae, ae_loss)
    n = toks.shape[0]
    print("training the program autoencoder ...", flush=True)
    for epoch in range(args.ae_epochs):
        perm = np.random.permutation(n)
        tot = 0.0
        nb = 0
        for k in range(0, n, args.bs):
            idx = perm[k:k + args.bs]
            lv, g = lg(mx.array(toks[idx]))
            opt.update(ae, g)
            mx.eval(ae.parameters(), opt.state, lv)
            tot += float(lv.item())
            nb += 1
        if (epoch + 1) % 10 == 0 or epoch == args.ae_epochs - 1:
            # reconstruction validity (greedy, teacher-free) — sanity, NOT the bar
            zr = ae.encode(mx.array(toks[:args.n_eval]))
            rec = measure(ar_decode(ae, zr, temperature=0.7), evaluate)
            print(f"AE epoch {epoch+1}/{args.ae_epochs} loss={tot/max(nb,1):.4f} "
                  f"recon_validity={rec['validity']:.3f} distinct={rec['num_distinct']}", flush=True)

    # encode the corpus, normalise the latent for diffusion
    print("encoding corpus -> latent bank ...", flush=True)
    zs = []
    for k in range(0, n, 512):
        zs.append(np.array(ae.encode(mx.array(toks[k:k + 512])).tolist()))
    zbank = np.concatenate(zs, axis=0).astype(np.float32)
    zmean = zbank.mean(0, keepdims=True)
    zstd = zbank.std(0, keepdims=True) + 1e-6
    zn = (zbank - zmean) / zstd
    zmean_mx, zstd_mx = mx.array(zmean), mx.array(zstd)

    # train the latent DDPM
    print("training the latent DDPM ...", flush=True)
    den = Denoiser(d_z=args.d_z)
    ddpm = DDPM(steps=args.ddpm_steps)
    dopt = optim.AdamW(learning_rate=3e-4, weight_decay=0.0)

    def dm_loss(z0):
        b = z0.shape[0]
        t_idx = mx.array(np.random.randint(0, args.ddpm_steps, b))
        noise = mx.random.normal(z0.shape)
        zt = ddpm.q_sample(z0, t_idx, noise)
        eps = den(zt, t_idx.astype(mx.float32) / args.ddpm_steps)
        return ((eps - noise) ** 2).mean()

    dlg = mlxnn.value_and_grad(den, dm_loss)
    m = zn.shape[0]
    for epoch in range(args.dm_epochs):
        perm = np.random.permutation(m)
        tot = 0.0
        nb = 0
        for k in range(0, m, args.bs):
            lv, g = dlg(mx.array(zn[perm[k:k + args.bs]]))
            dopt.update(den, g)
            mx.eval(den.parameters(), dopt.state, lv)
            tot += float(lv.item())
            nb += 1
        if (epoch + 1) % 50 == 0 or epoch == args.dm_epochs - 1:
            print(f"DDPM epoch {epoch+1}/{args.dm_epochs} loss={tot/max(nb,1):.5f}", flush=True)

    # === the acceptance bar: validity of DIFFUSION-SAMPLED z ===
    print("sampling z ~ diffusion -> decode -> execute ...", flush=True)
    zsamp = ddpm.sample(den, args.n_eval, args.d_z) * zstd_mx + zmean_mx
    sampled = measure(ar_decode(ae, zsamp, temperature=1.0), evaluate)
    # predict-the-mean baseline: z = 0 (normalised) -> unnormalise -> decode
    zzero = mx.broadcast_to(zmean_mx, (args.n_eval, args.d_z))
    zerob = measure(ar_decode(ae, zzero, temperature=1.0), evaluate)

    result = {"framework": "MLX", "model": "latent DDPM over a z-conditioned AR program autoencoder",
              "fix": "diffuse the construction-program latent + execution-respecting AR decoder "
                     "(replaces independent-face geometry diffusion that could not sew)",
              "n_train": int(n), "d_z": args.d_z, "ae_epochs": args.ae_epochs,
              "dm_epochs": args.dm_epochs, "ddpm_steps": args.ddpm_steps,
              "validity_gate": "is_solid AND volume>1e-4, real OCC kernel",
              "sampled_z_validity": round(sampled["validity"], 4), "sampled_z": sampled,
              "z0_mean_baseline_validity": round(zerob["validity"], 4), "z0_baseline": zerob,
              "checkpoint": f"{args.out}/latent_diffusion_mlx.safetensors"}
    mx.save_safetensors(f"{args.out}/latent_diffusion_mlx.safetensors",
                        dict(tree_flatten(ae.parameters())) | {f"den.{k}": v for k, v in tree_flatten(den.parameters())})
    with open(f"{args.out}/latent_diffusion_mlx_metrics.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print("LATENT_DIFFUSION_DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
