// Single source of truth for the MLX trainer script PATHS cited in runnable
// command snippets across the docs (e.g. `python ll_gen/mlx/ar_generator_mlx.py
// --mode train`). Injected at build time by `src/plugins/remark-metrics.mjs`,
// which replaces `{{script.<dotted.path>}}` tokens with the value below.
//
// Only the script PATH is centralized — not the `python` prefix or the `--mode`
// flag. The path is the thing that drifts when a script is renamed or moved; the
// flags are page-contextual (train vs parity) and stay literal per snippet.
// Change a path HERE and every runnable command that references it updates.
//
// NOTE: descriptive prose mentions of these filenames (e.g. "the generator
// (`ll_gen/mlx/ar_generator_mlx.py`) …") are intentionally left literal — they
// are single contextual descriptions, not copy-pasted command snippets.

export const scripts = {
  ll_gen: {
    arGenerator: 'll_gen/mlx/ar_generator_mlx.py',
    latentDiffusion: 'll_gen/mlx/latent_diffusion_mlx.py',
  },
  ll_brepnet: {
    train: 'll_brepnet/mlx/train_brepnet_mlx.py',
  },
  ll_ocadr: {
    faithfulTower: 'll_ocadr/mlx/faithful_tower_mlx.py',
    train: 'll_ocadr/mlx/train_ocadr_mlx.py',
  },
  ll_stepnet: {
    classification: 'll_stepnet/mlx/train_classification_mlx.py',
  },
};

export default scripts;
