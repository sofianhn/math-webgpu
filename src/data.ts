import type { Batch, DataPipeline, Dataset, TrainingStage } from "./types";
import { tensor } from "./tensor";

export const OPS = ["+", "-", "*", "/"] as const;
export type Op = (typeof OPS)[number];

/** Operands 0–9; answers mapped into classes 0..NUM_CLASSES-1 (clamped). */
export const MAX_OPERAND = 9;
/** Cover mul (81) and keep room; classes are answer values 0..NUM_CLASSES-1. */
export const NUM_CLASSES = 82;

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
 * Feature encoding: one-hot(a) + one-hot(op) + one-hot(b).
 * inputDim = 10 + 4 + 10 = 24. Small enough for a demo MLP.
 */
export const INPUT_DIM = MAX_OPERAND + 1 + OPS.length + MAX_OPERAND + 1;

export function encodeFeatures(a: number, op: Op, b: number): Float32Array {
  const x = new Float32Array(INPUT_DIM);
  x[a] = 1;
  x[MAX_OPERAND + 1 + OPS.indexOf(op)] = 1;
  x[MAX_OPERAND + 1 + OPS.length + b] = 1;
  return x;
}

export function formatExpr(a: number, op: Op, b: number, answer?: number): string {
  const sym = op === "*" ? "×" : op === "/" ? "÷" : op === "-" ? "−" : "+";
  const base = `${a} ${sym} ${b}`;
  return answer === undefined ? base : `${base} = ${answer}`;
}

/** Synthetic arithmetic dataset for pre-training (direct answer classification). */
export class ArithmeticDataset implements Dataset {
  readonly name = "synthetic-arithmetic";

  size(): number {
    return (MAX_OPERAND + 1) ** 2 * OPS.length;
  }

  sample(batchSize: number): Batch {
    const inputs = new Float32Array(batchSize * INPUT_DIM);
    const targets = new Int32Array(batchSize);
    const texts: string[] = [];

    for (let i = 0; i < batchSize; i++) {
      const a = Math.floor(Math.random() * (MAX_OPERAND + 1));
      const b = Math.floor(Math.random() * (MAX_OPERAND + 1));
      const op = OPS[Math.floor(Math.random() * OPS.length)];
      const answer = Math.min(NUM_CLASSES - 1, computeAnswer(a, op, b));
      inputs.set(encodeFeatures(a, op, b), i * INPUT_DIM);
      targets[i] = answer;
      texts.push(formatExpr(a, op, b, answer));
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
