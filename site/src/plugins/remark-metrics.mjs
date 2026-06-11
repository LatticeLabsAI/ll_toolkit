// Build-time data injection for the docs.
//
// Replaces `{{<namespace>.<dotted.path>}}` tokens in Markdown/MDX with the
// canonical value from the matching data module:
//   - `{{metric.…}}`  -> src/data/metrics.mjs   (published model numbers)
//   - `{{script.…}}`  -> src/data/scripts.mjs   (MLX trainer script paths)
//
// Runs on the mdast (before syntax highlighting / mdast->hast), so substitutions
// apply to prose, inline code, fenced code blocks, AND MDX JSX string attributes
// (e.g. `description="..."`).
//
// The separator after the namespace is a dot, NOT a colon: Starlight enables
// remark-directive (for `:::note` asides), which parses an inline `:name`
// sequence as a text directive and would split a `{{ns:…}}` token across several
// text nodes before this plugin runs. A dot is inert to inline parsing, so the
// whole token survives in one text node and matches cleanly.
//
// Fail-loud: a token whose path does not resolve to a scalar throws and fails
// the build, so a typo can never ship as a literal `{{…}}`.

import { metrics } from '../data/metrics.mjs';
import { scripts } from '../data/scripts.mjs';

// Maps a token namespace to its data root and source file (for error messages).
const REGISTRY = {
  metric: { root: metrics, source: 'src/data/metrics.mjs' },
  script: { root: scripts, source: 'src/data/scripts.mjs' },
};

// Matches e.g. `{{metric.ll_gen.ar.validity}}` or `{{script.ll_gen.arGenerator}}`
// with optional surrounding whitespace.
const TOKEN_RE = /\{\{\s*(metric|script)\.([A-Za-z0-9_.]+?)\s*\}\}/g;

function resolvePath(root, path) {
  return path
    .split('.')
    .reduce((obj, key) => (obj == null ? undefined : obj[key]), root);
}

function substitute(text, filePath) {
  return text.replace(TOKEN_RE, (match, namespace, path) => {
    const { root, source } = REGISTRY[namespace];
    const value = resolvePath(root, path);
    if (value === undefined || value === null || typeof value === 'object') {
      throw new Error(
        `remark-metrics: unresolved token "${match}" ` +
          `(namespace "${namespace}", path "${path}") in ${filePath ?? '<unknown file>'}. ` +
          `Add it to ${source} or fix the token.`,
      );
    }
    return String(value);
  });
}

// Recursive walk: unist-util-visit only traverses `children`, but MDX JSX
// attributes live on `node.attributes`, so we descend both. Any node carrying a
// string `value` (text, inlineCode, code, mdxJsxAttribute) is a substitution
// site; attribute-expression values are objects and are left untouched.
function walk(node, filePath) {
  if (node == null || typeof node !== 'object') return;
  if (typeof node.value === 'string') {
    node.value = substitute(node.value, filePath);
  }
  if (Array.isArray(node.attributes)) {
    for (const attr of node.attributes) walk(attr, filePath);
  }
  if (Array.isArray(node.children)) {
    for (const child of node.children) walk(child, filePath);
  }
}

export default function remarkMetrics() {
  return function transformer(tree, file) {
    walk(tree, file?.path);
  };
}
