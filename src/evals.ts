import type { Batch, Model } from "./types";
import { argmaxRow, crossEntropy, softmaxRows, tensor } from "./tensor";
import {
  INPUT_DIM,
  NUM_CLASSES,
  OPS,
  TOKENS_PER_EXAMPLE,
  computeAnswer,
  describeExample,
  encodeFeatures,
  type Op,
} from "./data";

export interface EvalReport {
  /** Mean cross-entropy (nats). */
  loss: number;
  /** exp(loss) — classifier perplexity (how “surprised” on average). */
  perplexity: number;
  accuracy: number;
  n: number;
  /** Rough token count evaluated (examples × tokens/example). */
  tokens: number;
}

/** Fixed held-out bank the trainer never samples from. */
export type HoldoutExpr = readonly [number, Op, number];

/**
 * Hold out every expression whose ones-digit on a or b is in `onesDigits`.
 * Training never sees these; evals measure generalization to held-out digit patterns.
 */
export function buildOnesDigitHoldout(onesDigits: number[], cap = 256): HoldoutExpr[] {
  const want = new Set(onesDigits);
  const out: HoldoutExpr[] = [];
  for (const op of OPS) {
    for (let a = 0; a <= 99 && out.length < cap; a++) {
      for (let b = 0; b <= 99 && out.length < cap; b++) {
        if (!want.has(a % 10) && !want.has(b % 10)) continue;
        const answer = computeAnswer(a, op, b);
        if (answer >= NUM_CLASSES) continue;
        out.push([a, op, b]);
      }
    }
  }
  // Deterministic shuffle (mulberry-ish)
  let s = 0xc0ffee;
  for (let i = out.length - 1; i > 0; i--) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const j = s % (i + 1);
    const tmp = out[i]!;
    out[i] = out[j]!;
    out[j] = tmp;
  }
  return out.slice(0, Math.min(cap, out.length));
}

export function isHeldOut(a: number, b: number, onesDigits: number[]): boolean {
  const want = new Set(onesDigits);
  return want.has(a % 10) || want.has(b % 10);
}

export function batchFromExprs(exprs: HoldoutExpr[]): Batch {
  const B = exprs.length;
  const inputs = new Float32Array(B * INPUT_DIM);
  const targets = new Int32Array(B);
  const texts: string[] = [];
  for (let i = 0; i < B; i++) {
    const [a, op, b] = exprs[i]!;
    const answer = computeAnswer(a, op, b);
    inputs.set(encodeFeatures(a, op, b), i * INPUT_DIM);
    targets[i] = answer;
    texts.push(describeExample(a, op, b, answer));
  }
  return { inputs: tensor([B, INPUT_DIM], inputs), targets, texts };
}

export async function evaluateModel(model: Model, batch: Batch): Promise<EvalReport> {
  const { logits } = await model.forward(batch.inputs);
  const [B, C] = logits.shape;
  const loss = crossEntropy(logits.data, B, C, batch.targets);
  const probs = softmaxRows(logits.data, B, C);
  const preds = argmaxRow(probs, B, C);
  let correct = 0;
  for (let i = 0; i < B; i++) if (preds[i] === batch.targets[i]) correct++;
  return {
    loss,
    perplexity: Math.exp(loss),
    accuracy: correct / B,
    n: B,
    tokens: B * TOKENS_PER_EXAMPLE,
  };
}

export { TOKENS_PER_EXAMPLE };
