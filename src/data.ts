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

/** Fixed curriculum probes (mix of 1- and 2-digit) for the smartness scoreboard. */
export const CURRICULUM: ReadonlyArray<readonly [number, Op, number]> = [
  [3, "+", 5],
  [12, "+", 34],
  [15, "*", 6],
  [50, "-", 17],
  [84, "/", 4],
  [7, "*", 7],
  [99, "+", 1],
  [40, "/", 8],
];

export type SampleFilter = (a: number, b: number, op: Op) => boolean;

function sampleInRange(filter?: SampleFilter): { a: number; op: Op; b: number; answer: number } {
  for (let attempt = 0; attempt < 80; attempt++) {
    const a = Math.floor(Math.random() * (MAX_OPERAND + 1));
    const b = Math.floor(Math.random() * (MAX_OPERAND + 1));
    const op = OPS[Math.floor(Math.random() * OPS.length)]!;
    if (filter && !filter(a, b, op)) continue;
    const answer = computeAnswer(a, op, b);
    if (answer < NUM_CLASSES) return { a, op, b, answer };
  }
  // Fallback that always fits and usually passes ones-digit holdouts that exclude 7.
  const a = Math.floor(Math.random() * 50);
  const b = Math.floor(Math.random() * 50);
  const op: Op = "+";
  if (!filter || filter(a, b, op)) return { a, op, b, answer: a + b };
  return { a: 0, op: "+", b: 0, answer: 0 };
}

/** Synthetic arithmetic dataset for pre-training (answer classification). */
export class ArithmeticDataset implements Dataset {
  readonly name = "synthetic-arithmetic-2digit";
  private readonly filter?: SampleFilter;

  constructor(filter?: SampleFilter) {
    this.filter = filter;
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
      const { a, op, b, answer } = sampleInRange(this.filter);
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
 * Stage-aware pipeline. pretrain → synthetic arithmetic.
 * finetune / rl hooks throw until implemented (keeps the plug-in surface clear).
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
    if (this.stage === "rl") {
      throw new Error("rl pipeline not implemented — plug preference / reward batches here.");
    }
    return this.dataset.sample(batchSize);
  }
}
