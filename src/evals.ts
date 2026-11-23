import type { Batch, Model } from "./types";
import { argmaxRow, crossEntropy, softmaxRows, tensor } from "./tensor";
import {
  INPUT_DIM,
  MAX_OPERAND,
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

/** Default: hold out ~10% of exact (a,op,b) triples — every digit 0–9 still appears in train. */
export const FAIR_HOLDOUT_PCT = 10;
export const FAIR_HOLDOUT_SEED = 0xc0ffee7;

/** Stable hash so the held-out bank is identical across runs. */
export function exprHash(a: number, op: Op, b: number, seed = FAIR_HOLDOUT_SEED): number {
  const oi = OPS.indexOf(op);
  let h = seed >>> 0;
  h = Math.imul(h ^ (a + 1), 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (oi + 1), 0xc2b2ae35) >>> 0;
  h = Math.imul(h ^ (b + 1), 0x27d4eb2d) >>> 0;
  return h >>> 0;
}

/** True when this exact expression is in the fair held-out slice. */
export function isExpressionHeldOut(
  a: number,
  op: Op,
  b: number,
  pct = FAIR_HOLDOUT_PCT,
): boolean {
  return exprHash(a, op, b) % 100 < pct;
}

/**
 * Freeze a bank of exact expressions for eval. Training still sees every digit
 * 0–9 on other expressions — unlike ones-digit holdout, which never activates
 * that input feature.
 */
export function buildExpressionHoldout(cap = 256, pct = FAIR_HOLDOUT_PCT): HoldoutExpr[] {
  const out: HoldoutExpr[] = [];
  for (const op of OPS) {
    for (let a = 0; a <= MAX_OPERAND; a++) {
      for (let b = 0; b <= MAX_OPERAND; b++) {
        if (!isExpressionHeldOut(a, op, b, pct)) continue;
        const answer = computeAnswer(a, op, b);
        if (answer >= NUM_CLASSES) continue;
        out.push([a, op, b]);
      }
    }
  }
  let s = FAIR_HOLDOUT_SEED;
  for (let i = out.length - 1; i > 0; i--) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const j = s % (i + 1);
    const tmp = out[i]!;
    out[i] = out[j]!;
    out[j] = tmp;
  }
  return out.slice(0, Math.min(cap, out.length));
}

/**
 * Harsh probe: expressions whose ones-digit on a or b is in `onesDigits`.
 * Optional classroom demo — do NOT use as the default train filter (that
 * feature channel never fires, so failure is almost guaranteed).
 */
export function buildOnesDigitHoldout(onesDigits: number[], cap = 256): HoldoutExpr[] {
  const want = new Set(onesDigits);
  const out: HoldoutExpr[] = [];
  for (const op of OPS) {
    for (let a = 0; a <= MAX_OPERAND && out.length < cap * 4; a++) {
      for (let b = 0; b <= MAX_OPERAND && out.length < cap * 4; b++) {
        if (!want.has(a % 10) && !want.has(b % 10)) continue;
        const answer = computeAnswer(a, op, b);
        if (answer >= NUM_CLASSES) continue;
        out.push([a, op, b]);
      }
    }
  }
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

export function isOnesDigitHeldOut(a: number, b: number, onesDigits: number[]): boolean {
  const want = new Set(onesDigits);
  return want.has(a % 10) || want.has(b % 10);
}

/** @deprecated use isOnesDigitHeldOut — kept for call-site clarity during migrate */
export function isHeldOut(a: number, b: number, onesDigits: number[]): boolean {
  return isOnesDigitHeldOut(a, b, onesDigits);
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
