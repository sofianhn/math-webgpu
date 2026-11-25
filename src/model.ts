import type { ForwardResult, Model, Tensor } from "./types";
import type { MatmulBackend } from "./webgpu";
import {
  activePlaceCount,
  addBiasInPlace,
  randn,
  reluInPlace,
  softmaxRows,
  tensor,
  zeros,
} from "./tensor";
import { ANSWER_DIGITS, INPUT_DIM, fromDigits } from "./data";

export const DEFAULT_HIDDEN = 256;
/** Softmax width per answer place. */
export const DIGIT_CLASSES = 10;

/**
 * Place-value MLP calculator: input → ReLU → ReLU → independent digit heads
 * (one 10-way classifier per answer place, LSD-first, 18 places).
 *
 * Independent heads matched the successful two-digit H/T/O design; a shared
 * head + place embedding stalled ~19% exact on the same domain.
 */
export class TinyMathMlp implements Model {
  readonly name = "digit-math-mlp-9";
  readonly inputDim = INPUT_DIM;
  readonly answerDigits = ANSWER_DIGITS;
  readonly hidden: number;

  private readonly gpu: MatmulBackend;
  private readonly W1: Tensor;
  private readonly b1: Tensor;
  private readonly W2: Tensor;
  private readonly b2: Tensor;
  /** Stacked per-place heads: row block p is W for place p. Shape [ANSWER_DIGITS * H, 10]. */
  private readonly Wheads: Tensor;
  /** Stacked biases: [ANSWER_DIGITS * 10]. */
  private readonly bheads: Tensor;

  constructor(gpu: MatmulBackend, hidden = DEFAULT_HIDDEN) {
    this.gpu = gpu;
    this.hidden = hidden;
    this.W1 = randn([INPUT_DIM, hidden], Math.sqrt(2 / INPUT_DIM));
    this.b1 = zeros([hidden]);
    this.W2 = randn([hidden, hidden], Math.sqrt(2 / hidden));
    this.b2 = zeros([hidden]);
    const scale = Math.sqrt(2 / hidden);
    this.Wheads = randn([ANSWER_DIGITS * hidden, DIGIT_CLASSES], scale);
    this.bheads = zeros([ANSWER_DIGITS * DIGIT_CLASSES]);
  }

  static paramCount(hidden: number): number {
    const trunk = INPUT_DIM * hidden + hidden + hidden * hidden + hidden;
    const heads = ANSWER_DIGITS * (hidden * DIGIT_CLASSES + DIGIT_CLASSES);
    return trunk + heads;
  }

  weightNorms(): { name: string; l2: number; count: number }[] {
    return this.parameters().map(({ name, tensor: t }) => {
      let s = 0;
      for (let i = 0; i < t.data.length; i++) s += t.data[i]! * t.data[i]!;
      return { name, l2: Math.sqrt(s), count: t.data.length };
    });
  }

  parameters() {
    return [
      { name: "W1", tensor: this.W1 },
      { name: "b1", tensor: this.b1 },
      { name: "W2", tensor: this.W2 },
      { name: "b2", tensor: this.b2 },
      { name: "Wheads", tensor: this.Wheads },
      { name: "bheads", tensor: this.bheads },
    ];
  }

  private async trunk(inputs: Tensor): Promise<{
    B: number;
    h1Pre: Float32Array;
    h1: Float32Array;
    h2Pre: Float32Array;
    h2: Float32Array;
  }> {
    const [B, D] = inputs.shape;
    const H = this.hidden;
    if (D !== INPUT_DIM) throw new Error(`expected inputDim ${INPUT_DIM}, got ${D}`);
    const h1Pre = await this.gpu.matmul(inputs.data, this.W1.data, B, INPUT_DIM, H);
    addBiasInPlace(h1Pre, B, H, this.b1.data);
    const h1 = h1Pre.slice();
    reluInPlace(h1);
    const h2Pre = await this.gpu.matmul(h1, this.W2.data, B, H, H);
    addBiasInPlace(h2Pre, B, H, this.b2.data);
    const h2 = h2Pre.slice();
    reluInPlace(h2);
    return { B, h1Pre, h1, h2Pre, h2 };
  }

  private digitLogits(h: Float32Array, B: number): Float32Array {
    const H = this.hidden;
    const logits = new Float32Array(B * ANSWER_DIGITS * DIGIT_CLASSES);
    for (let i = 0; i < B; i++) {
      for (let p = 0; p < ANSWER_DIGITS; p++) {
        const wOff = p * H * DIGIT_CLASSES;
        const bOff = p * DIGIT_CLASSES;
        const outOff = (i * ANSWER_DIGITS + p) * DIGIT_CLASSES;
        for (let d = 0; d < DIGIT_CLASSES; d++) {
          let s = this.bheads.data[bOff + d]!;
          for (let k = 0; k < H; k++) {
            s += h[i * H + k]! * this.Wheads.data[wOff + k * DIGIT_CLASSES + d]!;
          }
          logits[outOff + d] = s;
        }
      }
    }
    return logits;
  }

  async forward(inputs: Tensor): Promise<ForwardResult> {
    const { B, h1Pre, h1, h2Pre, h2 } = await this.trunk(inputs);
    const logits = this.digitLogits(h2, B);
    return {
      logits: tensor([B, ANSWER_DIGITS * DIGIT_CLASSES], logits),
      cache: {
        x: inputs,
        h1Pre: tensor([B, this.hidden], h1Pre),
        h1: tensor([B, this.hidden], h1),
        h2Pre: tensor([B, this.hidden], h2Pre),
        h: tensor([B, this.hidden], h2),
        digitLogits: tensor([B, ANSWER_DIGITS * DIGIT_CLASSES], logits),
      },
    };
  }

  backward(
    cache: Record<string, Tensor>,
    digitTargets: Uint8Array,
    activeCounts?: Uint8Array,
  ): Map<string, Tensor> {
    const x = cache.x!;
    const h1 = cache.h1!;
    const h1Pre = cache.h1Pre!;
    const h2 = cache.h!;
    const h2Pre = cache.h2Pre!;
    const digitLogits = cache.digitLogits!;
    const [B] = x.shape;
    const H = this.hidden;
    const rows = B * ANSWER_DIGITS;

    if (digitTargets.length !== rows) {
      throw new Error(`digitTargets length ${digitTargets.length} != ${rows}`);
    }

    const probs = softmaxRows(digitLogits.data, rows, DIGIT_CLASSES);
    const dLogits = new Float32Array(probs.length);
    for (let i = 0; i < B; i++) {
      const active = activeCounts
        ? Math.min(ANSWER_DIGITS, Math.max(1, activeCounts[i]!))
        : ANSWER_DIGITS;
      const rowDigits = digitTargets.subarray(i * ANSWER_DIGITS, (i + 1) * ANSWER_DIGITS);
      const sig = activePlaceCount(rowDigits, ANSWER_DIGITS);
      for (let p = 0; p < active; p++) {
        const row = i * ANSWER_DIGITS + p;
        const dig = digitTargets[row]!;
        const w = p < sig ? (p < 4 ? 1.35 : 1.15) : p < sig + 2 ? 0.55 : 0.18;
        for (let d = 0; d < DIGIT_CLASSES; d++) {
          dLogits[row * DIGIT_CLASSES + d] = (probs[row * DIGIT_CLASSES + d]! / B) * w;
        }
        dLogits[row * DIGIT_CLASSES + dig]! -= w / B;
      }
    }

    const dWheads = zeros([ANSWER_DIGITS * H, DIGIT_CLASSES]);
    const dbheads = zeros([ANSWER_DIGITS * DIGIT_CLASSES]);
    const dH2 = new Float32Array(B * H);

    for (let i = 0; i < B; i++) {
      for (let p = 0; p < ANSWER_DIGITS; p++) {
        const wOff = p * H * DIGIT_CLASSES;
        const bOff = p * DIGIT_CLASSES;
        const row = i * ANSWER_DIGITS + p;
        for (let k = 0; k < H; k++) {
          const hk = h2.data[i * H + k]!;
          let back = 0;
          for (let d = 0; d < DIGIT_CLASSES; d++) {
            const g = dLogits[row * DIGIT_CLASSES + d]!;
            dWheads.data[wOff + k * DIGIT_CLASSES + d]! += hk * g;
            back += g * this.Wheads.data[wOff + k * DIGIT_CLASSES + d]!;
          }
          dH2[i * H + k]! += back;
        }
        for (let d = 0; d < DIGIT_CLASSES; d++) {
          dbheads.data[bOff + d]! += dLogits[row * DIGIT_CLASSES + d]!;
        }
      }
    }

    const dH2Pre = dH2.slice();
    for (let i = 0; i < dH2Pre.length; i++) {
      if (h2Pre.data[i]! <= 0) dH2Pre[i] = 0;
    }

    const dW2 = zeros([H, H]);
    const db2 = zeros([H]);
    const dH1 = new Float32Array(B * H);
    for (let i = 0; i < B; i++) {
      for (let k = 0; k < H; k++) {
        const g = dH2Pre[i * H + k]!;
        db2.data[k]! += g;
        for (let j = 0; j < H; j++) {
          dW2.data[j * H + k]! += h1.data[i * H + j]! * g;
          dH1[i * H + j]! += g * this.W2.data[j * H + k]!;
        }
      }
    }

    const dH1Pre = dH1.slice();
    for (let i = 0; i < dH1Pre.length; i++) {
      if (h1Pre.data[i]! <= 0) dH1Pre[i] = 0;
    }

    const dW1 = zeros([INPUT_DIM, H]);
    const db1 = zeros([H]);
    for (let i = 0; i < B; i++) {
      for (let d = 0; d < INPUT_DIM; d++) {
        const xd = x.data[i * INPUT_DIM + d]!;
        for (let k = 0; k < H; k++) {
          dW1.data[d * H + k]! += xd * dH1Pre[i * H + k]!;
        }
      }
      for (let k = 0; k < H; k++) db1.data[k]! += dH1Pre[i * H + k]!;
    }

    return new Map([
      ["W1", dW1],
      ["b1", db1],
      ["W2", dW2],
      ["b2", db2],
      ["Wheads", dWheads],
      ["bheads", dbheads],
    ]);
  }

  applyGradients(grads: Map<string, Tensor>, lr: number): void {
    for (const { name, tensor: param } of this.parameters()) {
      const g = grads.get(name);
      if (!g) continue;
      for (let i = 0; i < param.data.length; i++) param.data[i]! -= lr * g.data[i]!;
    }
  }

  async predict(features: Float32Array): Promise<{
    answer: bigint;
    digits: number[];
    digitProbs: Float32Array;
  }> {
    const { logits } = await this.forward(tensor([1, INPUT_DIM], features));
    const probs = softmaxRows(logits.data, ANSWER_DIGITS, DIGIT_CLASSES);
    const digits: number[] = [];
    for (let p = 0; p < ANSWER_DIGITS; p++) {
      let best = 0;
      let bestV = probs[p * DIGIT_CLASSES]!;
      for (let d = 1; d < DIGIT_CLASSES; d++) {
        const v = probs[p * DIGIT_CLASSES + d]!;
        if (v > bestV) {
          bestV = v;
          best = d;
        }
      }
      digits.push(best);
    }
    return { answer: fromDigits(digits), digits, digitProbs: probs };
  }
}
