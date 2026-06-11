// Single source of truth for published model metrics cited across the docs.
//
// These are the headline numbers the narrative pages report. They are injected
// at build time by `src/plugins/remark-metrics.mjs`, which replaces
// `{{metric.<dotted.path>}}` tokens in any `.md` / `.mdx` page with the value
// below. Change a number HERE and every page that cites it updates; an unknown
// token fails the build loudly (same fail-loud philosophy as scripts/gen_api.py).
//
// Provenance: the ll_gen figures are measured through the real OCC/CadQuery
// kernel by the native-MLX trainers, gated on a non-degenerate solid:
//   - ar.*              `python ll_gen/mlx/ar_generator_mlx.py --mode train`
//   - latentDiffusion.* `python ll_gen/mlx/latent_diffusion_mlx.py --mode train`
// They are DeepCAD-trained results and change only on retrain. When they do,
// update this file (and only this file).

export const metrics = {
  ll_gen: {
    // Autoregressive command generator: causal transformer over the CAD command
    // vocabulary, sampled token-by-token then executed. Plain prior-sampling
    // validity ("measured validity").
    ar: {
      validity: '0.914',
      validFraction: '234/256',
      distinct: '104',
    },
    // Latent diffusion over a program autoencoder: sample z -> decode
    // autoregressively -> execute. The headline metric is SAMPLED-Z validity
    // (noise -> denoise -> decode -> execute), reported against a z=0
    // predict-the-mean baseline whose diverse-shape count is `baselineDistinct`.
    latentDiffusion: {
      sampledZValidity: '0.934',
      validFraction: '239/256',
      distinct: '138',
      baselineDistinct: '14',
    },
  },
};

export default metrics;
