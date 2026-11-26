import type { Batch, Model } from "./types";
import { digitAccuracy, digitCrossEntropy, tensor } from "./tensor";
import {
  ANSWER_DIGITS,
  INPUT_DIM,
  OPERAND_DIGITS,
  OPS,
  TOKENS_PER_EXAMPLE,
  computeAnswer,
  describeExample,
  digitLength,
  encodeFeatures,
  randomOperand,
  toDigits,
  type Op,
} from "./data";

export interface EvalReport {
  /** Mean digit cross-entropy (nats). */
  loss: number;
  /** exp(loss) — digit-classifier perplexity. */
  perplexity: number;
  /** Exact-match accuracy (all answer digits correct). */
  accuracy: number;
  /** Mean per-digit accuracy. */
  digitAccuracy: number;
  n: number;
  /** Rough token count evaluated (examples × tokens/example). */
  tokens: number;
}

/** Fixed held-out bank the trainer never samples from. */
export type HoldoutExpr = readonly [bigint, Op, bigint];

/** Default: hold out ~10% of exact (a,op,b) triples by hash. */
export const FAIR_HOLDOUT_PCT = 10;
export const FAIR_HOLDOUT_SEED = 0xc0ffee7;

/** Stable hash so the held-out bank is identical across runs. */
export function exprHash(a: bigint, op: Op, b: bigint, seed = FAIR_HOLDOUT_SEED): number {
  const oi = OPS.indexOf(op);
  let h = seed >>> 0;
  // Mix low 32 bits of each operand (enough for stable bucketing).
  const amix = Number(a % 0x100000000n);
  const bmix = Number(b % 0x100000000n);
  h = Math.imul(h ^ (amix + 1), 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (oi + 1), 0xc2b2ae35) >>> 0;
  h = Math.imul(h ^ (bmix + 1), 0x27d4eb2d) >>> 0;
  // Fold higher digits when present.
  if (a >= 0x100000000n) h = Math.imul(h ^ Number((a >> 32n) + 1n), 0x165667b1) >>> 0;
  if (b >= 0x100000000n) h = Math.imul(h ^ Number((b >> 32n) + 1n), 0x27d4eb2f) >>> 0;
  return h >>> 0;
}

/** True when this exact expression is in the fair held-out slice. */
export function isExpressionHeldOut(
  a: bigint,
  op: Op,
  b: bigint,
  pct = FAIR_HOLDOUT_PCT,
): boolean {
  return exprHash(a, op, b) % 100 < pct;
}

/**
 * Freeze a bank of exact expressions for eval across digit lengths.
 * Cannot enumerate 9-digit space — sample with a fixed LCG seed.
 */
export function buildExpressionHoldout(
  cap = 256,
  pct = FAIR_HOLDOUT_PCT,
  maxDigits = OPERAND_DIGITS,
): HoldoutExpr[] {
  const out: HoldoutExpr[] = [];
  let s = FAIR_HOLDOUT_SEED;
  const next = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s;
  };

  // Cover short lengths densely, then sample longer ones.
  for (let digits = 1; digits <= Math.min(3, maxDigits); digits++) {
    const hi = 10 ** digits;
    for (const op of OPS) {
      for (let a = 0; a < hi && out.length < cap * 4; a++) {
        for (let b = 0; b < hi && out.length < cap * 4; b++) {
          const aa = BigInt(a);
          const bb = BigInt(b);
          if (!isExpressionHeldOut(aa, op, bb, pct)) continue;
          const answer = computeAnswer(aa, op, bb);
          if (digitLength(answer) > ANSWER_DIGITS) continue;
          out.push([aa, op, bb]);
        }
      }
    }
  }

  /** Seeded operand with exactly `digits` places (holdout must be reproducible). */
  const seededOperand = (digits: number): bigint => {
    const d = Math.max(1, Math.min(OPERAND_DIGITS, digits));
    if (d === 1) return BigInt(next() % 10);
    const lo = 10n ** BigInt(d - 1);
    const span = 10n ** BigInt(d) - lo;
    // Mix two 32-bit draws for up to 9-digit spans.
    const r = BigInt(next()) + (BigInt(next()) << 32n);
    return lo + (r % span);
  };

  // Stochastic fill for longer curricula (4–9 digits).
  let guard = 0;
  while (out.length < cap * 3 && guard < cap * 80) {
    guard++;
    const digA = 1 + (next() % maxDigits);
    const digB = 1 + (next() % maxDigits);
    const a = seededOperand(digA);
    const b = seededOperand(digB);
    const op = OPS[next() % OPS.length]!;
    if (!isExpressionHeldOut(a, op, b, pct)) continue;
    const answer = computeAnswer(a, op, b);
    if (digitLength(answer) > ANSWER_DIGITS) continue;
    out.push([a, op, b]);
  }

  // Deterministic shuffle + cap.
  for (let i = out.length - 1; i > 0; i--) {
    const j = next() % (i + 1);
    const tmp = out[i]!;
    out[i] = out[j]!;
    out[j] = tmp;
  }
  return out.slice(0, Math.min(cap, out.length));
}

/**
 * Harsh probe: expressions whose ones-digit on a or b is in `onesDigits`.
 * Optional classroom demo — do NOT use as the default train filter.
 */
export function buildOnesDigitHoldout(onesDigits: number[], cap = 256): HoldoutExpr[] {
  const want = new Set(onesDigits);
  const out: HoldoutExpr[] = [];
  for (const op of OPS) {
    for (let a = 0; a <= 99 && out.length < cap * 4; a++) {
      for (let b = 0; b <= 99 && out.length < cap * 4; b++) {
        if (!want.has(a % 10) && !want.has(b % 10)) continue;
        out.push([BigInt(a), op, BigInt(b)]);
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

export function isOnesDigitHeldOut(a: bigint, b: bigint, onesDigits: number[]): boolean {
  const want = new Set(onesDigits);
  return want.has(Number(a % 10n)) || want.has(Number(b % 10n));
}

/** @deprecated use isOnesDigitHeldOut */
export function isHeldOut(a: bigint, b: bigint, onesDigits: number[]): boolean {
  return isOnesDigitHeldOut(a, b, onesDigits);
}

export function batchFromExprs(exprs: HoldoutExpr[], frontierDigits = OPERAND_DIGITS): Batch {
  const B = exprs.length;
  const inputs = new Float32Array(B * INPUT_DIM);
  const digitTargets = new Uint8Array(B * ANSWER_DIGITS);
  const activeCounts = new Uint8Array(B);
  const texts: string[] = [];
  const answers: string[] = [];
  void frontierDigits;
  for (let i = 0; i < B; i++) {
    const [a, op, b] = exprs[i]!;
    const answer = computeAnswer(a, op, b);
    inputs.set(encodeFeatures(a, op, b), i * INPUT_DIM);
    digitTargets.set(toDigits(answer, ANSWER_DIGITS), i * ANSWER_DIGITS);
    activeCounts[i] = ANSWER_DIGITS;
    texts.push(describeExample(a, op, b, answer));
    answers.push(answer.toString());
  }
  return { inputs: tensor([B, INPUT_DIM], inputs), digitTargets, activeCounts, texts, answers };
}

/** Stratified probes by max operand digit length. */
export function buildLengthProbes(
  maxDigits: number,
  perBucket = 32,
): { digits: number; exprs: HoldoutExpr[] }[] {
  const buckets: { digits: number; exprs: HoldoutExpr[] }[] = [];
  for (let d = 1; d <= maxDigits; d++) {
    const exprs: HoldoutExpr[] = [];
    let guard = 0;
    while (exprs.length < perBucket && guard < perBucket * 40) {
      guard++;
      const a = randomOperand(d);
      const b = randomOperand(1 + Math.floor(Math.random() * d));
      const op = OPS[Math.floor(Math.random() * OPS.length)]!;
      const answer = computeAnswer(a, op, b);
      if (digitLength(answer) > ANSWER_DIGITS) continue;
      exprs.push([a, op, b]);
    }
    buckets.push({ digits: d, exprs });
  }
  return buckets;
}

export async function evaluateModel(model: Model, batch: Batch): Promise<EvalReport> {
  const { logits } = await model.forward(batch.inputs);
  const [B] = logits.shape;
  const places = model.answerDigits;
  const loss = digitCrossEntropy(logits.data, B, places, batch.digitTargets, 10, batch.activeCounts);
  const { exact, digit } = digitAccuracy(
    logits.data,
    B,
    places,
    batch.digitTargets,
    10,
    batch.activeCounts,
  );
  return {
    loss,
    perplexity: Math.exp(loss),
    accuracy: exact,
    digitAccuracy: digit,
    n: B,
    tokens: B * TOKENS_PER_EXAMPLE,
  };
}

export { TOKENS_PER_EXAMPLE };
