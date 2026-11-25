import type { Batch, DataPipeline, Dataset, TrainingStage } from "./types";
import { tensor } from "./tensor";

export const OPS = ["+", "-", "*", "/"] as const;
export type Op = (typeof OPS)[number];

/** Max operand length the encoder supports (nine decimal digits). */
export const OPERAND_DIGITS = 9;
/** Max answer places (9×9 mul → up to 18 digits). */
export const ANSWER_DIGITS = 18;
/** One-hot width per place. */
export const DIGIT_SLOTS = 10;
/** Largest in-domain operand: 10^9 − 1. */
export const MAX_OPERAND = 10 ** OPERAND_DIGITS - 1; // 999_999_999

/**
 * Place-value one-hots for both operands + op:
 * 9×10 (a) + 4 (op) + 9×10 (b) = 184.
 */
export const INPUT_DIM =
  OPERAND_DIGITS * DIGIT_SLOTS + OPS.length + OPERAND_DIGITS * DIGIT_SLOTS;

/** Feature “tokens”: each place digit + the operator. */
export const TOKENS_PER_EXAMPLE = OPERAND_DIGITS * 2 + 1;

export function computeAnswer(a: bigint, op: Op, b: bigint): bigint {
  switch (op) {
    case "+":
      return a + b;
    case "-":
      return a >= b ? a - b : 0n;
    case "*":
      return a * b;
    case "/":
      return b === 0n ? 0n : a / b;
  }
}

/** Decimal digit count of a non-negative bigint (0 → 1). */
export function digitLength(n: bigint): number {
  if (n < 0n) throw new Error("digitLength expects non-negative");
  if (n === 0n) return 1;
  let len = 0;
  let x = n;
  while (x > 0n) {
    x /= 10n;
    len++;
  }
  return len;
}

/** LSD-first digits, zero-padded to `places`. */
export function toDigits(n: bigint, places: number = ANSWER_DIGITS): Uint8Array {
  if (n < 0n) throw new Error("toDigits expects non-negative");
  const out = new Uint8Array(places);
  let x = n;
  for (let p = 0; p < places; p++) {
    out[p] = Number(x % 10n);
    x /= 10n;
  }
  if (x !== 0n) throw new Error(`value ${n} does not fit in ${places} digits`);
  return out;
}

/** Recompose LSD-first digits into a bigint. */
export function fromDigits(digits: ArrayLike<number>, len?: number): bigint {
  const n = len ?? digits.length;
  let v = 0n;
  let place = 1n;
  for (let p = 0; p < n; p++) {
    const d = digits[p]! | 0;
    if (d < 0 || d > 9) throw new Error(`bad digit ${d}`);
    v += BigInt(d) * place;
    place *= 10n;
  }
  return v;
}

export function inDomain(a: bigint | number, b: bigint | number): boolean {
  const aa = typeof a === "bigint" ? a : BigInt(a);
  const bb = typeof b === "bigint" ? b : BigInt(b);
  return aa >= 0n && aa <= BigInt(MAX_OPERAND) && bb >= 0n && bb <= BigInt(MAX_OPERAND);
}

/** True when a value needs more than OPERAND_DIGITS places. */
export function isBeyondOperandDigits(n: bigint | number): boolean {
  const v = typeof n === "bigint" ? n : BigInt(Math.trunc(n));
  return v < 0n || digitLength(v) > OPERAND_DIGITS;
}

function encodeOperand(x: Float32Array, offset: number, value: bigint): void {
  if (value < 0n || value > BigInt(MAX_OPERAND)) return;
  let v = value;
  for (let place = 0; place < OPERAND_DIGITS; place++) {
    const dig = Number(v % 10n);
    x[offset + place * DIGIT_SLOTS + dig] = 1;
    v /= 10n;
  }
}

/**
 * Encode place-value features for up to 9-digit operands.
 * Out-of-range operands leave empty digit slots.
 */
export function encodeFeatures(a: bigint | number, op: Op, b: bigint | number): Float32Array {
  const aa = typeof a === "bigint" ? a : BigInt(a);
  const bb = typeof b === "bigint" ? b : BigInt(b);
  const x = new Float32Array(INPUT_DIM);
  encodeOperand(x, 0, aa);
  x[OPERAND_DIGITS * DIGIT_SLOTS + OPS.indexOf(op)] = 1;
  encodeOperand(x, OPERAND_DIGITS * DIGIT_SLOTS + OPS.length, bb);
  return x;
}

export function formatExpr(
  a: bigint | number | string,
  op: Op,
  b: bigint | number | string,
  answer?: bigint | number | string,
): string {
  const sym = op === "*" ? "×" : op === "/" ? "÷" : op === "-" ? "−" : "+";
  const base = `${a} ${sym} ${b}`;
  return answer === undefined ? base : `${base} = ${answer}`;
}

/** Human-readable description of one training example’s features. */
export function describeExample(a: bigint, op: Op, b: bigint, answer: bigint): string {
  const aDigs = [...toDigits(a, OPERAND_DIGITS)].reverse().join("");
  const bDigs = [...toDigits(b, OPERAND_DIGITS)].reverse().join("");
  return `${formatExpr(a, op, b, answer)}  ·  a=${aDigs} op=${op} b=${bDigs}  → ${ANSWER_DIGITS} digit heads`;
}

/** Fixed curriculum probes spanning short → long operands. */
export const CURRICULUM: ReadonlyArray<readonly [bigint, Op, bigint]> = [
  [3n, "+", 5n],
  [12n, "+", 34n],
  [15n, "*", 6n],
  [50n, "-", 17n],
  [84n, "/", 4n],
  [47n, "+", 38n],
  [17n, "*", 19n],
  [999n, "+", 1n],
  [1234n, "*", 7n],
  [99999n, "-", 12345n],
  [123456n, "+", 654321n],
  [1000000n, "/", 8n],
  [12345678n, "+", 87654321n],
  [999999999n, "-", 1n],
  [123456789n, "+", 987654321n],
  [111111111n, "*", 2n],
  [876543210n, "/", 9n],
  [314159265n, "+", 271828182n],
];

export type SampleFilter = (a: bigint, b: bigint, op: Op) => boolean;

/** How often each op appears. Ninja favors mul + carry-heavy add. */
export type OpMix = Record<Op, number>;

export const BALANCED_MIX: OpMix = { "+": 1, "-": 1, "*": 1, "/": 1 };

/** Empirically better for hard products / carries across digit lengths. */
export const NINJA_MIX: OpMix = { "+": 0.3, "-": 0.12, "*": 0.46, "/": 0.12 };

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

/** Uniform random integer in [0, maxInclusive]. */
function randBigIntInclusive(maxInclusive: bigint): bigint {
  if (maxInclusive <= 0n) return 0n;
  if (maxInclusive <= BigInt(Number.MAX_SAFE_INTEGER)) {
    return BigInt(Math.floor(Math.random() * (Number(maxInclusive) + 1)));
  }
  const bits = maxInclusive.toString(2).length;
  const bytes = Math.ceil(bits / 8);
  let v = 0n;
  for (let i = 0; i < bytes; i++) {
    v = (v << 8n) | BigInt(Math.floor(Math.random() * 256));
  }
  return v % (maxInclusive + 1n);
}

/** Random operand with exactly `digits` decimal places (or 0 when digits=1). */
export function randomOperand(digits: number): bigint {
  const d = Math.max(1, Math.min(OPERAND_DIGITS, Math.floor(digits)));
  if (d === 1) return randBigIntInclusive(9n);
  const lo = 10n ** BigInt(d - 1);
  const hi = 10n ** BigInt(d) - 1n;
  return lo + randBigIntInclusive(hi - lo);
}

/** Random operand with 1..maxDigits places (slight bias toward longer). */
export function randomOperandUpTo(maxDigits: number): bigint {
  const m = Math.max(1, Math.min(OPERAND_DIGITS, Math.floor(maxDigits)));
  let r = Math.random() * ((m * (m + 1)) / 2);
  let digits = 1;
  for (let k = 1; k <= m; k++) {
    r -= k;
    if (r <= 0) {
      digits = k;
      break;
    }
  }
  if (digits === 1 && Math.random() < 0.15) return 0n;
  return randomOperand(digits);
}

function onesDigit(n: bigint): number {
  return Number(n % 10n);
}

/** Carry / borrow / multi-digit-mul / non-trivial div hardness. */
function isHardCase(a: bigint, b: bigint, op: Op): boolean {
  if (op === "+") return onesDigit(a) + onesDigit(b) >= 10;
  if (op === "-") return onesDigit(a) < onesDigit(b);
  if (op === "*") return digitLength(a) >= 2 && digitLength(b) >= 2;
  return b > 1n && a % b !== 0n;
}

export type DatasetOptions = {
  filter?: SampleFilter;
  mix?: OpMix;
  hardBias?: number;
  maxDigits?: number;
};

/**
 * Synthetic arithmetic dataset for pre-training.
 * Samples across a digit-length curriculum up to nine digits; answers encoded
 * as 18 LSD-first digit heads (BigInt-safe).
 */
export class ArithmeticDataset implements Dataset {
  readonly name = "synthetic-arithmetic-9digit";
  private readonly filter?: SampleFilter;
  private readonly mix: OpMix;
  private hardBias: number;
  private maxDigits: number;

  constructor(filterOrOpts?: SampleFilter | DatasetOptions) {
    if (typeof filterOrOpts === "function") {
      this.filter = filterOrOpts;
      this.mix = BALANCED_MIX;
      this.hardBias = 0;
      this.maxDigits = OPERAND_DIGITS;
    } else {
      this.filter = filterOrOpts?.filter;
      this.mix = filterOrOpts?.mix ?? BALANCED_MIX;
      this.hardBias = filterOrOpts?.hardBias ?? 0;
      this.maxDigits = Math.max(1, Math.min(OPERAND_DIGITS, filterOrOpts?.maxDigits ?? OPERAND_DIGITS));
    }
  }

  setMaxDigits(maxDigits: number): void {
    this.maxDigits = Math.max(1, Math.min(OPERAND_DIGITS, Math.floor(maxDigits)));
  }

  getMaxDigits(): number {
    return this.maxDigits;
  }

  setHardBias(hardBias: number): void {
    this.hardBias = hardBias;
  }

  size(): number {
    return Number(10n ** BigInt(this.maxDigits)) ** 2 * OPS.length;
  }

  sample(batchSize: number): Batch {
    const inputs = new Float32Array(batchSize * INPUT_DIM);
    const digitTargets = new Uint8Array(batchSize * ANSWER_DIGITS);
    const activeCounts = new Uint8Array(batchSize);
    const texts: string[] = [];
    const answers: string[] = [];
    for (let i = 0; i < batchSize; i++) {
      const { a, op, b, answer } = this.sampleOne();
      inputs.set(encodeFeatures(a, op, b), i * INPUT_DIM);
      digitTargets.set(toDigits(answer, ANSWER_DIGITS), i * ANSWER_DIGITS);
      // Always supervise every answer place so high zeros stay zero; loss weights
      // down-weight far trailing zeros (see model.backward / digitCrossEntropy).
      activeCounts[i] = ANSWER_DIGITS;
      texts.push(describeExample(a, op, b, answer));
      answers.push(answer.toString());
    }

    return {
      inputs: tensor([batchSize, INPUT_DIM], inputs),
      digitTargets,
      activeCounts,
      texts,
      answers,
    };
  }

  private sampleOne(): { a: bigint; op: Op; b: bigint; answer: bigint } {
    const maxD = this.maxDigits;
    for (let attempt = 0; attempt < 160; attempt++) {
      const preferHard = this.hardBias > 0 && Math.random() < this.hardBias;
      const op = pickOp(this.mix);
      let a = randomOperandUpTo(maxD);
      let b = randomOperandUpTo(maxD);

      if (preferHard) {
        if (op === "*") {
          const da = 1 + Math.floor(Math.random() * maxD);
          const maxDb = Math.min(maxD, ANSWER_DIGITS - da);
          const db = Math.max(1, maxDb);
          a = randomOperand(da);
          b = randomOperand(Math.min(db, Math.max(1, Math.ceil(da / 2) + Math.floor(Math.random() * db))));
          if (maxD >= 2 && digitLength(a) === 1) a = randomOperand(2);
          if (maxD >= 2 && digitLength(b) === 1) b = randomOperand(2);
        } else if (op === "+") {
          a = randomOperandUpTo(maxD);
          b = randomOperandUpTo(maxD);
          const ao = onesDigit(a);
          const need = 10 - ao;
          if (need <= 9) {
            const bo = need + Math.floor(Math.random() * (10 - need));
            b = b - BigInt(onesDigit(b)) + BigInt(Math.min(9, bo));
          }
        } else if (op === "-") {
          a = randomOperandUpTo(maxD);
          b = randomOperandUpTo(Math.min(maxD, digitLength(a)));
          if (b > a) {
            const tmp = a;
            a = b;
            b = tmp;
          }
        } else if (op === "/") {
          b = randomOperandUpTo(Math.min(maxD, 4));
          if (b === 0n) b = 1n + randBigIntInclusive(8n);
          if (Math.random() < 0.55) {
            const q = randomOperandUpTo(Math.max(1, maxD - digitLength(b) + 1));
            a = q * b;
            if (digitLength(a) > maxD) a = randomOperandUpTo(maxD);
          } else {
            a = randomOperandUpTo(maxD);
          }
        }
      }

      if (digitLength(a) > maxD || digitLength(b) > maxD) continue;
      if (this.filter && !this.filter(a, b, op)) continue;
      const answer = computeAnswer(a, op, b);
      if (digitLength(answer) > ANSWER_DIGITS) continue;
      if (preferHard && !isHardCase(a, b, op) && attempt < 100) continue;
      return { a, op, b, answer };
    }
    const a = randomOperand(Math.min(2, maxD));
    const b = randomOperand(Math.min(2, maxD));
    return { a, op: "+", b, answer: a + b };
  }
}

/**
 * Stage-aware pipeline. pretrain / rl → synthetic arithmetic.
 * finetune throws until instruction packs are plugged in.
 */
export class MathDataPipeline implements DataPipeline {
  readonly stage: TrainingStage;
  private readonly dataset: Dataset;

  constructor(stage: TrainingStage, dataset: Dataset = new ArithmeticDataset()) {
    this.stage = stage;
    this.dataset = dataset;
  }

  getDataset(): Dataset {
    return this.dataset;
  }

  nextBatch(batchSize: number): Batch {
    if (this.stage === "finetune") {
      throw new Error("finetune pipeline not implemented — plug instruction/answer packs here.");
    }
    return this.dataset.sample(batchSize);
  }
}
