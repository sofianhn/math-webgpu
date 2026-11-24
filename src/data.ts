import type { Batch, DataPipeline, Dataset, TrainingStage } from "./types";
import { tensor } from "./tensor";

export const OPS = ["+", "-", "*", "/"] as const;
export type Op = (typeof OPS)[number];

/**
 * Two-digit operands 0–99 (default curriculum).
 * Three-digit values (≥100) are out of distribution.
 */
export const MAX_OPERAND = 99;
/** Max answer class (mul is resampled until answer fits). */
export const NUM_CLASSES = 1000;

const DIGIT_SLOTS = 10; // tens or ones place: digits 0–9

export function computeAnswer(a: number, op: Op, b: number): number {
  switch (op) {
    case "+":
      return a + b;
    case "-":
      return Math.max(0, a - b);
    case "*":
      return a * b;
    case "/":
      return b === 0 ? 0 : Math.floor(a / b);
  }
}

/**
 * Place-value one-hots: tens(a)+ones(a)+op+tens(b)+ones(b).
 * inputDim = 10+10+4+10+10 = 44.
 * Unlike a flat one-hot over 0–99, this shares digit features across numbers
 * (e.g. 23 and 27 share “tens=2”) — still not a true arithmetic algorithm.
 */
export const INPUT_DIM = DIGIT_SLOTS + DIGIT_SLOTS + OPS.length + DIGIT_SLOTS + DIGIT_SLOTS;

/** Each expression is 5 place-value “tokens”: a_tens, a_ones, op, b_tens, b_ones. */
export const TOKENS_PER_EXAMPLE = 5;

export function inDomain(a: number, b: number): boolean {
  return (
    Number.isInteger(a) &&
    Number.isInteger(b) &&
    a >= 0 &&
    a <= MAX_OPERAND &&
    b >= 0 &&
    b <= MAX_OPERAND
  );
}

/** True when a value needs 3+ digits (or is negative / non-integer). */
export function isThreeDigitOrBeyond(n: number): boolean {
  return !Number.isInteger(n) || n < 0 || n > MAX_OPERAND;
}

function encodeOperand(x: Float32Array, offset: number, value: number): void {
  if (value < 0 || value > MAX_OPERAND) return;
  const tens = Math.floor(value / 10);
  const ones = value % 10;
  x[offset + tens] = 1;
  x[offset + DIGIT_SLOTS + ones] = 1;
}

/**
 * Encode place-value features. Out-of-range operands leave empty digit slots —
 * the net never trained on 3-digit numbers, so those predictions are meaningless.
 */
export function encodeFeatures(a: number, op: Op, b: number): Float32Array {
  const x = new Float32Array(INPUT_DIM);
  encodeOperand(x, 0, a);
  x[DIGIT_SLOTS * 2 + OPS.indexOf(op)] = 1;
  encodeOperand(x, DIGIT_SLOTS * 2 + OPS.length, b);
  return x;
}

export function formatExpr(a: number, op: Op, b: number, answer?: number): string {
  const sym = op === "*" ? "×" : op === "/" ? "÷" : op === "-" ? "−" : "+";
  const base = `${a} ${sym} ${b}`;
  return answer === undefined ? base : `${base} = ${answer}`;
}

/** Human-readable description of one training example’s features. */
export function describeExample(a: number, op: Op, b: number, answer: number): string {
  const at = Math.floor(a / 10);
  const ao = a % 10;
  const bt = Math.floor(b / 10);
  const bo = b % 10;
  return `${formatExpr(a, op, b, answer)}  ·  features a=${at}|${ao} op=${op} b=${bt}|${bo}  → class ${answer}`;
}

/** Fixed curriculum probes — easy + hard (carry / mul) for the smartness scoreboard. */
export const CURRICULUM: ReadonlyArray<readonly [number, Op, number]> = [
  [3, "+", 5],
  [12, "+", 34],
  [15, "*", 6],
  [50, "-", 17],
  [84, "/", 4],
  [7, "*", 7],
  [47, "+", 38],
  [17, "*", 19],
  [99, "+", 99],
  [25, "*", 25],
];

export type SampleFilter = (a: number, b: number, op: Op) => boolean;

/** How often each op appears. Ninja favors mul + carry-heavy add. */
export type OpMix = Record<Op, number>;

export const BALANCED_MIX: OpMix = { "+": 1, "-": 1, "*": 1, "/": 1 };

/** Empirically better for hard two-digit products / carries. */
export const NINJA_MIX: OpMix = { "+": 0.28, "-": 0.12, "*": 0.48, "/": 0.12 };

function pickOp(mix: OpMix): Op {
  const entries = OPS.map((op) => [op, mix[op]] as const);
  const total = entries.reduce((s, [, w]) => s + w, 0);
  let r = Math.random() * total;
  for (const [op, w] of entries) {
    r -= w;
    if (r <= 0) return op;
  }
  return "*";
}

/** Ones-digit carry on addition, or both operands ≥10 for mul — harder cases. */
function isHardCase(a: number, b: number, op: Op): boolean {
  if (op === "+") return (a % 10) + (b % 10) >= 10;
  if (op === "*") return a >= 10 && b >= 10 && a * b < NUM_CLASSES;
  if (op === "-") return a % 10 < b % 10; // borrow
  return b > 1 && a % b !== 0; // non-trivial div
}

function sampleInRange(
  filter?: SampleFilter,
  mix: OpMix = BALANCED_MIX,
  hardBias = 0,
): { a: number; op: Op; b: number; answer: number } {
  for (let attempt = 0; attempt < 120; attempt++) {
    const preferHard = hardBias > 0 && Math.random() < hardBias;
    let a = Math.floor(Math.random() * (MAX_OPERAND + 1));
    let b = Math.floor(Math.random() * (MAX_OPERAND + 1));
    const op = pickOp(mix);
    if (preferHard) {
      if (op === "*") {
        a = 10 + Math.floor(Math.random() * 90);
        b = 10 + Math.floor(Math.random() * 90);
        if (a * b >= NUM_CLASSES) {
          b = Math.max(10, Math.floor(999 / a));
        }
      } else if (op === "+") {
        const ao = Math.floor(Math.random() * 10);
        const bo = Math.floor(Math.random() * (10 - ao)) + (9 - ao); // force carry often
        a = Math.floor(Math.random() * 10) * 10 + ao;
        b = Math.floor(Math.random() * 10) * 10 + Math.min(9, bo);
      }
    }
    if (filter && !filter(a, b, op)) continue;
    const answer = computeAnswer(a, op, b);
    if (answer >= NUM_CLASSES) continue;
    if (preferHard && !isHardCase(a, b, op) && attempt < 80) continue;
    return { a, op, b, answer };
  }
  // Fallback that always fits class range; still respects filter when possible.
  for (let a = 0; a <= 20; a++) {
    for (let b = 0; b <= 20; b++) {
      const op: Op = "+";
      if (!filter || filter(a, b, op)) return { a, op, b, answer: a + b };
    }
  }
  return { a: 0, op: "+", b: 0, answer: 0 };
}

export type DatasetOptions = {
  filter?: SampleFilter;
  /** Op sampling weights. */
  mix?: OpMix;
  /** Probability of forcing a hard (carry / two-digit mul) draw. */
  hardBias?: number;
};

/** Synthetic arithmetic dataset for pre-training (answer classification). */
export class ArithmeticDataset implements Dataset {
  readonly name = "synthetic-arithmetic-2digit";
  private readonly filter?: SampleFilter;
  private readonly mix: OpMix;
  private readonly hardBias: number;

  constructor(filterOrOpts?: SampleFilter | DatasetOptions) {
    if (typeof filterOrOpts === "function") {
      this.filter = filterOrOpts;
      this.mix = BALANCED_MIX;
      this.hardBias = 0;
    } else {
      this.filter = filterOrOpts?.filter;
      this.mix = filterOrOpts?.mix ?? BALANCED_MIX;
      this.hardBias = filterOrOpts?.hardBias ?? 0;
    }
  }

  size(): number {
    // Approximate: most (a,op,b) fit; mul with huge products is resampled away.
    return (MAX_OPERAND + 1) ** 2 * OPS.length;
  }

  sample(batchSize: number): Batch {
    const inputs = new Float32Array(batchSize * INPUT_DIM);
    const targets = new Int32Array(batchSize);
    const texts: string[] = [];

    for (let i = 0; i < batchSize; i++) {
      const { a, op, b, answer } = sampleInRange(this.filter, this.mix, this.hardBias);
      inputs.set(encodeFeatures(a, op, b), i * INPUT_DIM);
      targets[i] = answer;
      texts.push(describeExample(a, op, b, answer));
    }

    return {
      inputs: tensor([batchSize, INPUT_DIM], inputs),
      targets,
      texts,
    };
  }
}

/**
 * Stage-aware pipeline. pretrain / rl → synthetic arithmetic batches
 * (rl rewards are applied in `src/posttrain.ts`, not here).
 * finetune still throws until instruction packs land.
 */
export class MathDataPipeline implements DataPipeline {
  readonly stage: TrainingStage;
  private readonly dataset: Dataset;

  constructor(stage: TrainingStage, dataset: Dataset = new ArithmeticDataset()) {
    this.stage = stage;
    this.dataset = dataset;
  }

  /** Expose the underlying dataset for UI size estimates. */
  getDataset(): Dataset {
    return this.dataset;
  }

  nextBatch(batchSize: number): Batch {
    if (this.stage === "finetune") {
      throw new Error("finetune pipeline not implemented — plug instruction/answer packs here.");
    }
    // pretrain + rl both draw arithmetic rows; digit-RL shapes the loss in posttrain.
    return this.dataset.sample(batchSize);
  }
}
