/**
 * Post-training without English chain-of-thought.
 *
 * Digit RL — sample answer digits, reward digit-wise correctness (+ exact bonus),
 * self-critical REINFORCE vs greedy baseline, light CE mix.
 * DPO-digits — prefer true digit sequence over a near-miss corruption.
 * Self-consistency — temperature sample + majority vote at inference (no update).
 */
import type { DataPipeline, Model, Tensor, TrainingStage } from "./types";
import { DIGIT_CLASSES, TinyMathMlp } from "./model";
import { softmaxRows, tensor } from "./tensor";
import {
  ANSWER_DIGITS,
  ArithmeticDataset,
  MathDataPipeline,
  NINJA_MIX,
  OPERAND_DIGITS,
  TOKENS_PER_EXAMPLE,
  fromDigits,
  type Op,
} from "./data";
import {
  FAIR_HOLDOUT_PCT,
  batchFromExprs,
  buildExpressionHoldout,
  evaluateModel,
  isExpressionHeldOut,
  type EvalReport,
  type HoldoutExpr,
} from "./evals";

export type PostTrainKind = "digit-rl" | "dpo-digits" | "self-consistency";

export interface PostTrainConfig {
  kind: PostTrainKind;
  learningRate?: number;
  steps?: number;
  batchSize?: number;
  /** Softmax temperature for digit sampling (1 = raw probs). */
  temperature?: number;
  /** Extra reward when all answer digits match. */
  exactBonus?: number;
  /** Fraction of CE-toward-truth mixed into each RL step. */
  ceMix?: number;
  /** DPO β temperature. */
  dpoBeta?: number;
  /** Self-consistency sample count. */
  voteSamples?: number;
  hardBias?: number;
  maxDigits?: number;
  excludeHoldout?: boolean;
  holdoutExprs?: HoldoutExpr[];
  onStep?: (m: DigitRlStepMetrics) => void | Promise<void>;
}

export interface DigitRlStepMetrics {
  step: number;
  steps: number;
  meanReward: number;
  meanDigitAcc: number;
  exactRate: number;
  greedyDigitAcc: number;
  baseline: number;
  meanAdvantage: number;
  kind?: PostTrainKind;
}

export interface DigitRlResult {
  kind: PostTrainKind;
  steps: number;
  history: DigitRlStepMetrics[];
  beforeHoldout: EvalReport;
  afterHoldout: EvalReport;
  holdoutDelta: number;
  meanReward: number;
}

export function describePostTrain(kind: PostTrainKind): string {
  switch (kind) {
    case "digit-rl":
      return "Reward exact (or digit-wise) correctness across 18 answer places — no English needed.";
    case "dpo-digits":
      return "Prefer correct answer digits over near-misses for the same expression.";
    case "self-consistency":
      return "Majority-vote digit heads / temperature samples at inference.";
  }
}

export function postTrainStage(kind: PostTrainKind): TrainingStage {
  if (kind === "self-consistency") return "pretrain";
  return "rl";
}

function argmaxDigit(probs: Float32Array, offset: number): number {
  let best = 0;
  let bestV = probs[offset]!;
  for (let d = 1; d < DIGIT_CLASSES; d++) {
    const v = probs[offset + d]!;
    if (v > bestV) {
      bestV = v;
      best = d;
    }
  }
  return best;
}

function sampleDigit(probs: Float32Array, offset: number, temperature: number): number {
  if (temperature <= 1e-6) return argmaxDigit(probs, offset);
  let sum = 0;
  const scaled = new Float32Array(DIGIT_CLASSES);
  for (let d = 0; d < DIGIT_CLASSES; d++) {
    const p = Math.max(probs[offset + d]!, 1e-12);
    const v = Math.pow(p, 1 / temperature);
    scaled[d] = v;
    sum += v;
  }
  let r = Math.random() * sum;
  for (let d = 0; d < DIGIT_CLASSES; d++) {
    r -= scaled[d]!;
    if (r <= 0) return d;
  }
  return DIGIT_CLASSES - 1;
}

/** Digit-wise reward in [0, 1 + exactBonus] over ANSWER_DIGITS places. */
export function digitRewardFromTargets(
  pred: Uint8Array | ArrayLike<number>,
  truth: Uint8Array | ArrayLike<number>,
  places = ANSWER_DIGITS,
  exactBonus = 0.5,
): { reward: number; digitAcc: number; exact: boolean } {
  let hits = 0;
  for (let p = 0; p < places; p++) {
    if (pred[p] === truth[p]) hits++;
  }
  const digitAcc = hits / places;
  const exact = hits === places;
  return { reward: digitAcc + (exact ? exactBonus : 0), digitAcc, exact };
}

function addGrads(into: Map<string, Tensor>, add: Map<string, Tensor>, scale: number): void {
  if (scale === 0) return;
  for (const [name, g] of add) {
    const dst = into.get(name);
    if (!dst) {
      const copy = { shape: g.shape, data: g.data.slice() };
      for (let i = 0; i < copy.data.length; i++) copy.data[i]! *= scale;
      into.set(name, copy);
      continue;
    }
    for (let i = 0; i < dst.data.length; i++) dst.data[i]! += g.data[i]! * scale;
  }
}

function gradsNonZero(grads: Map<string, Tensor>): boolean {
  for (const g of grads.values()) {
    for (let i = 0; i < g.data.length; i++) {
      if (g.data[i]! !== 0) return true;
    }
  }
  return false;
}

function adamStep(
  model: Model,
  grads: Map<string, Tensor>,
  lr: number,
  m: Map<string, Float32Array>,
  v: Map<string, Float32Array>,
  t: number,
): void {
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  const lrT = (lr * Math.sqrt(1 - beta2 ** t)) / (1 - beta1 ** t);
  for (const { name, tensor: param } of model.parameters()) {
    const g = grads.get(name);
    if (!g) continue;
    if (!m.has(name)) {
      m.set(name, new Float32Array(param.data.length));
      v.set(name, new Float32Array(param.data.length));
    }
    const mt = m.get(name)!;
    const vt = v.get(name)!;
    for (let i = 0; i < param.data.length; i++) {
      const gi = g.data[i]!;
      mt[i] = beta1 * mt[i]! + (1 - beta1) * gi;
      vt[i] = beta2 * vt[i]! + (1 - beta2) * gi * gi;
      param.data[i]! -= (lrT * mt[i]!) / (Math.sqrt(vt[i]!) + eps);
    }
  }
}

function rlPipeline(
  excludeHoldout: boolean,
  hardBias: number,
  maxDigits: number,
): DataPipeline {
  const filter = excludeHoldout
    ? (a: bigint, b: bigint, op: Op) => !isExpressionHeldOut(a, op, b)
    : undefined;
  return new MathDataPipeline(
    "rl",
    new ArithmeticDataset({ filter, mix: NINJA_MIX, hardBias, maxDigits }),
  );
}

function sliceDigits(src: Uint8Array, i: number, places: number): Uint8Array {
  return src.subarray(i * places, (i + 1) * places);
}

/**
 * Self-critical digit RL on a loaded model.
 */
export async function runDigitRl(
  model: TinyMathMlp,
  config: Omit<PostTrainConfig, "kind"> = {},
): Promise<DigitRlResult> {
  const steps = Math.max(1, config.steps ?? 80);
  const batchSize = Math.max(4, config.batchSize ?? 32);
  const learningRate = config.learningRate ?? 0.001;
  const temperature = config.temperature ?? 1.0;
  const exactBonus = config.exactBonus ?? 0.5;
  const ceMix = Math.min(0.9, Math.max(0, config.ceMix ?? 0.35));
  const hardBias = config.hardBias ?? 0.55;
  const maxDigits = config.maxDigits ?? OPERAND_DIGITS;
  const excludeHoldout = config.excludeHoldout !== false;
  const holdExprs =
    config.holdoutExprs ?? buildExpressionHoldout(256, FAIR_HOLDOUT_PCT, maxDigits);
  const pipeline = rlPipeline(excludeHoldout, hardBias, maxDigits);
  const places = model.answerDigits;

  const beforeHoldout = await evaluateModel(model, batchFromExprs(holdExprs));

  const history: DigitRlStepMetrics[] = [];
  const m = new Map<string, Float32Array>();
  const v = new Map<string, Float32Array>();
  let tAdam = 0;
  let baseline = beforeHoldout.accuracy;
  let rewardSum = 0;

  for (let step = 1; step <= steps; step++) {
    const batch = pipeline.nextBatch(batchSize);
    const { cache, logits } = await model.forward(batch.inputs);
    const B = batchSize;
    const probs = softmaxRows(logits.data, B * places, DIGIT_CLASSES);

    const sampledTargets = new Uint8Array(B * places);
    let rewardAcc = 0;
    let digitAccAcc = 0;
    let exactAcc = 0;
    let greedyDigitAccAcc = 0;
    let advAcc = 0;

    for (let i = 0; i < B; i++) {
      const truth = sliceDigits(batch.digitTargets, i, places);
      const greedy = new Uint8Array(places);
      const sampled = new Uint8Array(places);
      for (let p = 0; p < places; p++) {
        const off = (i * places + p) * DIGIT_CLASSES;
        greedy[p] = argmaxDigit(probs, off);
        sampled[p] = sampleDigit(probs, off, temperature);
      }
      sampledTargets.set(sampled, i * places);
      const greedyR = digitRewardFromTargets(greedy, truth, places, exactBonus);
      const sampleR = digitRewardFromTargets(sampled, truth, places, exactBonus);
      advAcc += sampleR.reward - greedyR.reward;
      rewardAcc += sampleR.reward;
      digitAccAcc += sampleR.digitAcc;
      exactAcc += sampleR.exact ? 1 : 0;
      greedyDigitAccAcc += greedyR.digitAcc;
    }

    const meanReward = rewardAcc / B;
    const meanAdv = advAcc / B;
    baseline = 0.9 * baseline + 0.1 * meanReward;
    rewardSum += meanReward;

    const combined = new Map<string, Tensor>();
    addGrads(combined, model.backward(cache, sampledTargets, batch.activeCounts), -meanAdv);
    if (ceMix > 0) {
      addGrads(combined, model.backward(cache, batch.digitTargets, batch.activeCounts), ceMix);
    }

    if (gradsNonZero(combined)) {
      tAdam += 1;
      adamStep(model, combined, learningRate, m, v, tAdam);
    }

    const metrics: DigitRlStepMetrics = {
      step,
      steps,
      meanReward,
      meanDigitAcc: digitAccAcc / B,
      exactRate: exactAcc / B,
      greedyDigitAcc: greedyDigitAccAcc / B,
      baseline,
      meanAdvantage: meanAdv,
      kind: "digit-rl",
    };
    history.push(metrics);
    await config.onStep?.(metrics);
  }

  const afterHoldout = await evaluateModel(model, batchFromExprs(holdExprs));
  return {
    kind: "digit-rl",
    steps,
    history,
    beforeHoldout,
    afterHoldout,
    holdoutDelta: afterHoldout.accuracy - beforeHoldout.accuracy,
    meanReward: rewardSum / steps,
  };
}

/**
 * DPO-style preference on digit sequences: chosen = truth, rejected = one-digit flip.
 * Uses CE(chosen) − CE(rejected) as a surrogate preference gradient.
 */
export async function runDpoDigits(
  model: TinyMathMlp,
  config: Omit<PostTrainConfig, "kind"> = {},
): Promise<DigitRlResult> {
  const steps = Math.max(1, config.steps ?? 60);
  const batchSize = Math.max(4, config.batchSize ?? 32);
  const learningRate = config.learningRate ?? 0.0008;
  const hardBias = config.hardBias ?? 0.5;
  const maxDigits = config.maxDigits ?? OPERAND_DIGITS;
  const excludeHoldout = config.excludeHoldout !== false;
  const holdExprs =
    config.holdoutExprs ?? buildExpressionHoldout(256, FAIR_HOLDOUT_PCT, maxDigits);
  const pipeline = rlPipeline(excludeHoldout, hardBias, maxDigits);
  const places = model.answerDigits;

  const beforeHoldout = await evaluateModel(model, batchFromExprs(holdExprs));
  const history: DigitRlStepMetrics[] = [];
  const m = new Map<string, Float32Array>();
  const v = new Map<string, Float32Array>();
  let tAdam = 0;
  let rewardSum = 0;

  for (let step = 1; step <= steps; step++) {
    const batch = pipeline.nextBatch(batchSize);
    const { cache, logits } = await model.forward(batch.inputs);
    const B = batchSize;
    const probs = softmaxRows(logits.data, B * places, DIGIT_CLASSES);

    const rejected = batch.digitTargets.slice();
    let digitAccAcc = 0;
    let exactAcc = 0;
    for (let i = 0; i < B; i++) {
      // Flip a random place that matters (bias toward lower places).
      const p = Math.floor(Math.random() * Math.min(places, 12));
      const idx = i * places + p;
      const cur = rejected[idx]!;
      rejected[idx] = (cur + 1 + Math.floor(Math.random() * 9)) % 10;
      const greedy = new Uint8Array(places);
      for (let q = 0; q < places; q++) {
        greedy[q] = argmaxDigit(probs, (i * places + q) * DIGIT_CLASSES);
      }
      const truth = sliceDigits(batch.digitTargets, i, places);
      const r = digitRewardFromTargets(greedy, truth, places, 0.5);
      digitAccAcc += r.digitAcc;
      exactAcc += r.exact ? 1 : 0;
    }

    // Push toward chosen, away from rejected.
    const combined = new Map<string, Tensor>();
    addGrads(combined, model.backward(cache, batch.digitTargets, batch.activeCounts), 1.0);
    addGrads(combined, model.backward(cache, rejected, batch.activeCounts), -0.7);

    if (gradsNonZero(combined)) {
      tAdam += 1;
      adamStep(model, combined, learningRate, m, v, tAdam);
    }

    const meanReward = digitAccAcc / B + 0.5 * (exactAcc / B);
    rewardSum += meanReward;
    const metrics: DigitRlStepMetrics = {
      step,
      steps,
      meanReward,
      meanDigitAcc: digitAccAcc / B,
      exactRate: exactAcc / B,
      greedyDigitAcc: digitAccAcc / B,
      baseline: beforeHoldout.accuracy,
      meanAdvantage: 0,
      kind: "dpo-digits",
    };
    history.push(metrics);
    await config.onStep?.(metrics);
  }

  const afterHoldout = await evaluateModel(model, batchFromExprs(holdExprs));
  return {
    kind: "dpo-digits",
    steps,
    history,
    beforeHoldout,
    afterHoldout,
    holdoutDelta: afterHoldout.accuracy - beforeHoldout.accuracy,
    meanReward: rewardSum / steps,
  };
}

/** Majority-vote self-consistency over temperature samples (inference only). */
export async function selfConsistentPredict(
  model: TinyMathMlp,
  features: Float32Array,
  samples = 8,
  temperature = 0.8,
): Promise<{ answer: bigint; digits: number[]; votes: number[] }> {
  const places = model.answerDigits;
  const tallies = Array.from({ length: places }, () => new Float32Array(DIGIT_CLASSES));

  for (let s = 0; s < samples; s++) {
    const { logits } = await model.forward(tensor([1, features.length], features));
    const probs = softmaxRows(logits.data, places, DIGIT_CLASSES);
    for (let p = 0; p < places; p++) {
      const dig =
        s === 0
          ? argmaxDigit(probs, p * DIGIT_CLASSES)
          : sampleDigit(probs, p * DIGIT_CLASSES, temperature);
      tallies[p]![dig]! += 1;
    }
  }

  const digits: number[] = [];
  const votes: number[] = [];
  for (let p = 0; p < places; p++) {
    let best = 0;
    let bestV = tallies[p]![0]!;
    for (let d = 1; d < DIGIT_CLASSES; d++) {
      if (tallies[p]![d]! > bestV) {
        bestV = tallies[p]![d]!;
        best = d;
      }
    }
    digits.push(best);
    votes.push(bestV);
  }
  return { answer: fromDigits(digits), digits, votes };
}

export async function runPostTrain(
  model: Model,
  config: PostTrainConfig,
): Promise<DigitRlResult> {
  if (!(model instanceof TinyMathMlp)) {
    throw new Error("post-train requires TinyMathMlp (digit heads).");
  }
  if (config.kind === "digit-rl") return runDigitRl(model, config);
  if (config.kind === "dpo-digits") return runDpoDigits(model, config);
  if (config.kind === "self-consistency") {
    throw new Error(
      "self-consistency is inference-only — call selfConsistentPredict(), not runPostTrain().",
    );
  }
  throw new Error(`unknown post-train kind`);
}

export { TOKENS_PER_EXAMPLE };
