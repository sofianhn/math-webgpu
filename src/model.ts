import type { ForwardResult, Model, Tensor } from "./types";
import type { MatmulBackend } from "./webgpu";
import { addBiasInPlace, randn, reluInPlace, softmaxRows, tensor, zeros } from "./tensor";
import { ANSWER_DIGITS, INPUT_DIM, fromDigits } from "./data";

export const DEFAULT_HIDDEN = 256;
/** Softmax width per answer place. */
export const DIGIT_CLASSES = 10;
/** Place embedding width (shared digit head). */
export const PLACE_DIM = 32;

/**
 * Place-value MLP calculator: input → ReLU → ReLU → shared digit head
 * conditioned on place embeddings (18 LSD-first answer places).
 *
 * Shared head + place emb generalizes across digit lengths better than
 * independent H/T/O heads once operands stretch to nine digits.
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
  /** [ANSWER_DIGITS, PLACE_DIM] place embeddings. */
  private readonly placeEmb: Tensor;
  /** Shared digit head: [H + PLACE_DIM, 10]. */
  private readonly Wd: Tensor;
  private readonly bd: Tensor;

  constructor(gpu: MatmulBackend, hidden = DEFAULT_HIDDEN) {
    this.gpu = gpu;
    this.hidden = hidden;
    this.W1 = randn([INPUT_DIM, hidden], Math.sqrt(2 / INPUT_DIM));
    this.b1 = zeros([hidden]);
    this.W2 = randn([hidden, hidden], Math.sqrt(2 / hidden));
    this.b2 = zeros([hidden]);
    this.placeEmb = randn([ANSWER_DIGITS, PLACE_DIM], 0.02);
    const inD = hidden + PLACE_DIM;
    this.Wd = randn([inD, DIGIT_CLASSES], Math.sqrt(2 / inD));
    this.bd = zeros([DIGIT_CLASSES]);
  }

  static paramCount(hidden: number): number {
    const trunk = INPUT_DIM * hidden + hidden + hidden * hidden + hidden;
    const place = ANSWER_DIGITS * PLACE_DIM;
    const head = (hidden + PLACE_DIM) * DIGIT_CLASSES + DIGIT_CLASSES;
    return trunk + place + head;
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
      { name: "placeEmb", tensor: this.placeEmb },
      { name: "Wd", tensor: this.Wd },
      { name: "bd", tensor: this.bd },
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

  /** Digit logits [B, ANSWER_DIGITS * 10] and per-row head inputs [B*ANSWER_DIGITS, H+PLACE_DIM]. */
  private digitLogits(h: Float32Array, B: number): {
    logits: Float32Array;
    headIn: Float32Array;
  } {
    const H = this.hidden;
    const inD = H + PLACE_DIM;
    const logits = new Float32Array(B * ANSWER_DIGITS * DIGIT_CLASSES);
    const headIn = new Float32Array(B * ANSWER_DIGITS * inD);

    for (let i = 0; i < B; i++) {
      for (let p = 0; p < ANSWER_DIGITS; p++) {
        const row = (i * ANSWER_DIGITS + p) * inD;
        for (let k = 0; k < H; k++) headIn[row + k] = h[i * H + k]!;
        for (let e = 0; e < PLACE_DIM; e++) {
          headIn[row + H + e] = this.placeEmb.data[p * PLACE_DIM + e]!;
        }
        const outOff = (i * ANSWER_DIGITS + p) * DIGIT_CLASSES;
        for (let d = 0; d < DIGIT_CLASSES; d++) {
          let s = this.bd.data[d]!;
          for (let k = 0; k < inD; k++) {
            s += headIn[row + k]! * this.Wd.data[k * DIGIT_CLASSES + d]!;
          }
          logits[outOff + d] = s;
        }
      }
    }
    return { logits, headIn };
  }

  async forward(inputs: Tensor): Promise<ForwardResult> {
    const { B, h1Pre, h1, h2Pre, h2 } = await this.trunk(inputs);
    const { logits, headIn } = this.digitLogits(h2, B);
    const inD = this.hidden + PLACE_DIM;

    return {
      logits: tensor([B, ANSWER_DIGITS * DIGIT_CLASSES], logits),
      cache: {
        x: inputs,
        h1Pre: tensor([B, this.hidden], h1Pre),
        h1: tensor([B, this.hidden], h1),
        h2Pre: tensor([B, this.hidden], h2Pre),
        h: tensor([B, this.hidden], h2),
        headIn: tensor([B * ANSWER_DIGITS, inD], headIn),
        digitLogits: tensor([B, ANSWER_DIGITS * DIGIT_CLASSES], logits),
      },
    };
  }

  backward(cache: Record<string, Tensor>, digitTargets: Uint8Array): Map<string, Tensor> {
    const x = cache.x!;
    const h1 = cache.h1!;
    const h1Pre = cache.h1Pre!;
    const h2 = cache.h!;
    const h2Pre = cache.h2Pre!;
    const headIn = cache.headIn!;
    const digitLogits = cache.digitLogits!;
    const [B] = x.shape;
    const H = this.hidden;
    const inD = H + PLACE_DIM;
    const rows = B * ANSWER_DIGITS;

    if (digitTargets.length !== rows) {
      throw new Error(`digitTargets length ${digitTargets.length} != ${rows}`);
    }

    const probs = softmaxRows(digitLogits.data, rows, DIGIT_CLASSES);
    const dLogits = probs.slice();
    // Place weights: lower places (carries) slightly heavier; mid mul places matter too.
    for (let i = 0; i < B; i++) {
      for (let p = 0; p < ANSWER_DIGITS; p++) {
        const row = i * ANSWER_DIGITS + p;
        const dig = digitTargets[row]!;
        dLogits[row * DIGIT_CLASSES + dig]! -= 1;
        const w = p < 6 ? 1.25 : p < 12 ? 1.1 : 0.85;
        for (let d = 0; d < DIGIT_CLASSES; d++) {
          dLogits[row * DIGIT_CLASSES + d]! = (dLogits[row * DIGIT_CLASSES + d]! / B) * w;
        }
      }
    }

    const dWd = zeros([inD, DIGIT_CLASSES]);
    const dbd = zeros([DIGIT_CLASSES]);
    const dHeadIn = new Float32Array(rows * inD);
    const dPlace = zeros([ANSWER_DIGITS, PLACE_DIM]);
    const dH2 = new Float32Array(B * H);

    for (let row = 0; row < rows; row++) {
      const p = row % ANSWER_DIGITS;
      const i = Math.floor(row / ANSWER_DIGITS);
      for (let k = 0; k < inD; k++) {
        const hk = headIn.data[row * inD + k]!;
        let back = 0;
        for (let d = 0; d < DIGIT_CLASSES; d++) {
          const g = dLogits[row * DIGIT_CLASSES + d]!;
          dWd.data[k * DIGIT_CLASSES + d]! += hk * g;
          back += g * this.Wd.data[k * DIGIT_CLASSES + d]!;
        }
        dHeadIn[row * inD + k] = back;
      }
      for (let d = 0; d < DIGIT_CLASSES; d++) {
        dbd.data[d]! += dLogits[row * DIGIT_CLASSES + d]!;
      }
      for (let k = 0; k < H; k++) {
        dH2[i * H + k]! += dHeadIn[row * inD + k]!;
      }
      for (let e = 0; e < PLACE_DIM; e++) {
        dPlace.data[p * PLACE_DIM + e]! += dHeadIn[row * inD + H + e]!;
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

    void h2; // kept for API symmetry / future residual hooks
    return new Map([
      ["W1", dW1],
      ["b1", db1],
      ["W2", dW2],
      ["b2", db2],
      ["placeEmb", dPlace],
      ["Wd", dWd],
      ["bd", dbd],
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
