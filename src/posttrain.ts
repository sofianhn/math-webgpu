/**
 * Post-training without English chain-of-thought.
 *
 * The MLP has no language channel — only place-value features → digit heads.
 * Digit RL samples H/T/O from the policy, scores digit-wise correctness
 * (+ exact-match bonus), and updates with self-critical REINFORCE (sample
 * vs greedy baseline), plus a light CE mix toward the true answer so a
 * strong pretrained checkpoint is not destroyed.
 */
import type { DataPipeline, Model, Tensor, TrainingStage } from "./types";
import { DIGIT_CLASSES, TinyMathMlp } from "./model";
import { softmaxRows } from "./tensor";
import {
  ArithmeticDataset,
  MathDataPipeline,
  NINJA_MIX,
  TOKENS_PER_EXAMPLE,
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
  /** Extra reward when all three digits match. */
  exactBonus?: number;
  /** Fraction of CE-toward-truth mixed into each step (0 = pure REINFORCE). */
  ceMix?: number;
  /** Op mix hard-bias for RL batches (favor harder arithmetic). */
  hardBias?: number;
  /** Exclude fair held-out expressions from RL batches (default true). */
  excludeHoldout?: boolean;
  /** Optional frozen bank for before/after eval. */
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
}

export interface DigitRlResult {
  kind: "digit-rl";
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
      return "Reward exact (or digit-wise) correctness — no English needed.";
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

/** Digit-wise reward in [0, 1 + exactBonus]. */
export function digitReward(
  pred: number,
  truth: number,
  exactBonus = 0.5,
): { reward: number; digitAcc: number; exact: boolean } {
  const ph = Math.floor(pred / 100);
  const pt = Math.floor(pred / 10) % 10;
  const po = pred % 10;
  const th = Math.floor(truth / 100);
  const tt = Math.floor(truth / 10) % 10;
  const to = truth % 10;
  const hits = (ph === th ? 1 : 0) + (pt === tt ? 1 : 0) + (po === to ? 1 : 0);
  const digitAcc = hits / 3;
  const exact = pred === truth;
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

function rlPipeline(excludeHoldout: boolean, hardBias: number): DataPipeline {
  const filter = excludeHoldout
    ? (a: number, b: number, op: Op) => !isExpressionHeldOut(a, op, b)
    : undefined;
  return new MathDataPipeline(
    "rl",
    new ArithmeticDataset({ filter, mix: NINJA_MIX, hardBias }),
  );
}

/**
 * Self-critical digit RL on a loaded model.
 * Returns hold-out metrics before and after the update loop.
 */
export async function runDigitRl(
  model: TinyMathMlp,
  config: Omit<PostTrainConfig, "kind"> = {},
): Promise<DigitRlResult> {
  const steps = Math.max(1, config.steps ?? 80);
  const batchSize = Math.max(4, config.batchSize ?? 32);
  const learningRate = config.learningRate ?? 0.0015;
  const temperature = config.temperature ?? 1.0;
  const exactBonus = config.exactBonus ?? 0.5;
  const ceMix = Math.min(0.9, Math.max(0, config.ceMix ?? 0.35));
  const hardBias = config.hardBias ?? 0.55;
  const excludeHoldout = config.excludeHoldout !== false;
  const holdExprs = config.holdoutExprs ?? buildExpressionHoldout(200, FAIR_HOLDOUT_PCT);
  const pipeline = rlPipeline(excludeHoldout, hardBias);

  const beforeHoldout = await evaluateModel(model, batchFromExprs(holdExprs));

  const history: DigitRlStepMetrics[] = [];
  const m = new Map<string, Float32Array>();
  const v = new Map<string, Float32Array>();
  let tAdam = 0;
  let baseline = beforeHoldout.accuracy;
  let rewardSum = 0;

  for (let step = 1; step <= steps; step++) {
    const batch = pipeline.nextBatch(batchSize);
    const { cache } = await model.forward(batch.inputs);
    const B = batch.targets.length;
    const hundreds = cache.hundreds!;
    const tens = cache.tens!;
    const ones = cache.ones!;

    const ph = softmaxRows(hundreds.data, B, DIGIT_CLASSES);
    const pt = softmaxRows(tens.data, B, DIGIT_CLASSES);
    const po = softmaxRows(ones.data, B, DIGIT_CLASSES);

    const sampledTargets = new Int32Array(B);
    let rewardAcc = 0;
    let digitAccAcc = 0;
    let exactAcc = 0;
    let greedyDigitAccAcc = 0;
    let advAcc = 0;

    for (let i = 0; i < B; i++) {
      const truth = batch.targets[i]!;
      const off = i * DIGIT_CLASSES;
      const greedyAns =
        argmaxDigit(ph, off) * 100 + argmaxDigit(pt, off) * 10 + argmaxDigit(po, off);
      const greedyR = digitReward(greedyAns, truth, exactBonus);

      const sampleAns =
        sampleDigit(ph, off, temperature) * 100 +
        sampleDigit(pt, off, temperature) * 10 +
        sampleDigit(po, off, temperature);
      const sampleR = digitReward(sampleAns, truth, exactBonus);

      sampledTargets[i] = sampleAns;
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

    // REINFORCE: CE(sample) grads increase π(sample). Descend (−meanAdv)·CE
    // ≡ ascend meanAdv·log π(sample). Mix CE(truth) so pretrained quality holds.
    const combined = new Map<string, Tensor>();
    addGrads(combined, model.backward(cache, sampledTargets), -meanAdv);
    if (ceMix > 0) {
      addGrads(combined, model.backward(cache, batch.targets), ceMix);
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
 * Entry point for UI / blog. Currently implements digit-rl only;
 * other kinds throw with a descriptive message.
 */
export async function runPostTrain(
  model: Model,
  config: PostTrainConfig,
): Promise<DigitRlResult> {
  if (config.kind !== "digit-rl") {
    throw new Error(
      `post-train "${config.kind}" not implemented yet — ${describePostTrain(config.kind)}`,
    );
  }
  if (!(model instanceof TinyMathMlp)) {
    throw new Error("digit RL requires TinyMathMlp (digit heads).");
  }
  return runDigitRl(model, config);
}

export { TOKENS_PER_EXAMPLE };
