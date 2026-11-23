import type { ForwardResult, Model, Tensor } from "./types";
import type { MatmulBackend } from "./webgpu";
import { addBiasInPlace, randn, reluInPlace, softmaxRows, tensor, zeros } from "./tensor";
import { INPUT_DIM, NUM_CLASSES } from "./data";

export const DEFAULT_HIDDEN = 96;
/** Digits for hundreds / tens / ones of the answer (0…999). */
export const DIGIT_CLASSES = 10;

/**
 * Place-value MLP: shared trunk → three digit heads (hundreds, tens, ones).
 * The answer is composed as 100·H + 10·T + O instead of a flat 1000-way table.
 * That pushes the net toward digit structure instead of memorizing class IDs.
 *
 * Forward still exposes [batch, 1000] log-prob logits (product of digit
 * softmaxes) so the trainer/UI keep working; backward uses digit CE.
 */
export class TinyMathMlp implements Model {
  readonly name = "digit-math-mlp";
  readonly inputDim = INPUT_DIM;
  readonly numClasses = NUM_CLASSES;
  readonly hidden: number;

  private readonly gpu: MatmulBackend;
  private readonly W1: Tensor;
  private readonly b1: Tensor;
  /** Hundreds / tens / ones heads. */
  private readonly Wh: Tensor;
  private readonly bh: Tensor;
  private readonly Wt: Tensor;
  private readonly bt: Tensor;
  private readonly Wo: Tensor;
  private readonly bo: Tensor;

  constructor(gpu: MatmulBackend, hidden = DEFAULT_HIDDEN) {
    this.gpu = gpu;
    this.hidden = hidden;
    this.W1 = randn([INPUT_DIM, hidden], Math.sqrt(2 / INPUT_DIM));
    this.b1 = zeros([hidden]);
    const scale = Math.sqrt(2 / hidden);
    this.Wh = randn([hidden, DIGIT_CLASSES], scale);
    this.bh = zeros([DIGIT_CLASSES]);
    this.Wt = randn([hidden, DIGIT_CLASSES], scale);
    this.bt = zeros([DIGIT_CLASSES]);
    this.Wo = randn([hidden, DIGIT_CLASSES], scale);
    this.bo = zeros([DIGIT_CLASSES]);
  }

  /** Parameter count for the digit-head architecture. */
  static paramCount(hidden: number): number {
    const trunk = INPUT_DIM * hidden + hidden;
    const head = hidden * DIGIT_CLASSES + DIGIT_CLASSES;
    return trunk + 3 * head;
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
      { name: "Wh", tensor: this.Wh },
      { name: "bh", tensor: this.bh },
      { name: "Wt", tensor: this.Wt },
      { name: "bt", tensor: this.bt },
      { name: "Wo", tensor: this.Wo },
      { name: "bo", tensor: this.bo },
    ];
  }

  private async trunk(inputs: Tensor): Promise<{
    B: number;
    hPre: Float32Array;
    h: Float32Array;
  }> {
    const [B, D] = inputs.shape;
    const H = this.hidden;
    if (D !== INPUT_DIM) throw new Error(`expected inputDim ${INPUT_DIM}, got ${D}`);
    const hPreData = await this.gpu.matmul(inputs.data, this.W1.data, B, INPUT_DIM, H);
    addBiasInPlace(hPreData, B, H, this.b1.data);
    const hData = hPreData.slice();
    reluInPlace(hData);
    return { B, hPre: hPreData, h: hData };
  }

  private digitLogits(h: Float32Array, B: number): {
    hundreds: Float32Array;
    tens: Float32Array;
    ones: Float32Array;
  } {
    const H = this.hidden;
    const hundreds = new Float32Array(B * DIGIT_CLASSES);
    const tens = new Float32Array(B * DIGIT_CLASSES);
    const ones = new Float32Array(B * DIGIT_CLASSES);
    for (let i = 0; i < B; i++) {
      for (let d = 0; d < DIGIT_CLASSES; d++) {
        let sh = this.bh.data[d]!;
        let st = this.bt.data[d]!;
        let so = this.bo.data[d]!;
        for (let k = 0; k < H; k++) {
          const hk = h[i * H + k]!;
          sh += hk * this.Wh.data[k * DIGIT_CLASSES + d]!;
          st += hk * this.Wt.data[k * DIGIT_CLASSES + d]!;
          so += hk * this.Wo.data[k * DIGIT_CLASSES + d]!;
        }
        hundreds[i * DIGIT_CLASSES + d] = sh;
        tens[i * DIGIT_CLASSES + d] = st;
        ones[i * DIGIT_CLASSES + d] = so;
      }
    }
    return { hundreds, tens, ones };
  }

  /** Compose independent digit softmaxes into [B, 1000] log-probs (valid after softmax). */
  private composeClassLogits(
    ph: Float32Array,
    pt: Float32Array,
    po: Float32Array,
    B: number,
  ): Float32Array {
    const out = new Float32Array(B * NUM_CLASSES);
    for (let i = 0; i < B; i++) {
      for (let n = 0; n < NUM_CLASSES; n++) {
        const hd = Math.floor(n / 100);
        const td = Math.floor(n / 10) % 10;
        const od = n % 10;
        const p =
          ph[i * DIGIT_CLASSES + hd]! *
          pt[i * DIGIT_CLASSES + td]! *
          po[i * DIGIT_CLASSES + od]!;
        out[i * NUM_CLASSES + n] = Math.log(Math.max(p, 1e-12));
      }
    }
    return out;
  }

  async forward(inputs: Tensor): Promise<ForwardResult> {
    const { B, hPre, h } = await this.trunk(inputs);
    const { hundreds, tens, ones } = this.digitLogits(h, B);
    const ph = softmaxRows(hundreds, B, DIGIT_CLASSES);
    const pt = softmaxRows(tens, B, DIGIT_CLASSES);
    const po = softmaxRows(ones, B, DIGIT_CLASSES);
    const logitsData = this.composeClassLogits(ph, pt, po, B);

    return {
      logits: tensor([B, NUM_CLASSES], logitsData),
      cache: {
        x: inputs,
        hPre: tensor([B, this.hidden], hPre),
        h: tensor([B, this.hidden], h),
        hundreds: tensor([B, DIGIT_CLASSES], hundreds),
        tens: tensor([B, DIGIT_CLASSES], tens),
        ones: tensor([B, DIGIT_CLASSES], ones),
        logits: tensor([B, NUM_CLASSES], logitsData),
      },
    };
  }

  backward(cache: Record<string, Tensor>, targets: Int32Array): Map<string, Tensor> {
    const x = cache.x!;
    const h = cache.h!;
    const hPre = cache.hPre!;
    const hundreds = cache.hundreds!;
    const tens = cache.tens!;
    const ones = cache.ones!;
    const [B] = x.shape;
    const H = this.hidden;

    const ph = softmaxRows(hundreds.data, B, DIGIT_CLASSES);
    const pt = softmaxRows(tens.data, B, DIGIT_CLASSES);
    const po = softmaxRows(ones.data, B, DIGIT_CLASSES);

    const dH = new Float32Array(B * H);
    const dWh = zeros([H, DIGIT_CLASSES]);
    const dbh = zeros([DIGIT_CLASSES]);
    const dWt = zeros([H, DIGIT_CLASSES]);
    const dbt = zeros([DIGIT_CLASSES]);
    const dWo = zeros([H, DIGIT_CLASSES]);
    const dbo = zeros([DIGIT_CLASSES]);

    const accumHead = (
      probs: Float32Array,
      W: Tensor,
      dW: Tensor,
      db: Tensor,
      digitOf: (t: number) => number,
    ) => {
      const dLogits = probs.slice();
      for (let i = 0; i < B; i++) {
        const dig = digitOf(targets[i]!);
        dLogits[i * DIGIT_CLASSES + dig]! -= 1;
      }
      for (let i = 0; i < dLogits.length; i++) dLogits[i]! /= B;

      for (let i = 0; i < B; i++) {
        for (let k = 0; k < H; k++) {
          const hk = h.data[i * H + k]!;
          let back = 0;
          for (let d = 0; d < DIGIT_CLASSES; d++) {
            const g = dLogits[i * DIGIT_CLASSES + d]!;
            dW.data[k * DIGIT_CLASSES + d]! += hk * g;
            back += g * W.data[k * DIGIT_CLASSES + d]!;
          }
          dH[i * H + k]! += back;
        }
        for (let d = 0; d < DIGIT_CLASSES; d++) {
          db.data[d]! += dLogits[i * DIGIT_CLASSES + d]!;
        }
      }
    };

    accumHead(ph, this.Wh, dWh, dbh, (t) => Math.floor(t / 100));
    accumHead(pt, this.Wt, dWt, dbt, (t) => Math.floor(t / 10) % 10);
    accumHead(po, this.Wo, dWo, dbo, (t) => t % 10);

    const dHPre = dH.slice();
    for (let i = 0; i < dHPre.length; i++) {
      if (hPre.data[i]! <= 0) dHPre[i] = 0;
    }

    const dW1 = zeros([INPUT_DIM, H]);
    const db1 = zeros([H]);
    for (let i = 0; i < B; i++) {
      for (let d = 0; d < INPUT_DIM; d++) {
        const xd = x.data[i * INPUT_DIM + d]!;
        for (let k = 0; k < H; k++) {
          dW1.data[d * H + k]! += xd * dHPre[i * H + k]!;
        }
      }
      for (let k = 0; k < H; k++) db1.data[k]! += dHPre[i * H + k]!;
    }

    return new Map([
      ["W1", dW1],
      ["b1", db1],
      ["Wh", dWh],
      ["bh", dbh],
      ["Wt", dWt],
      ["bt", dbt],
      ["Wo", dWo],
      ["bo", dbo],
    ]);
  }

  applyGradients(grads: Map<string, Tensor>, lr: number): void {
    for (const { name, tensor: param } of this.parameters()) {
      const g = grads.get(name);
      if (!g) continue;
      for (let i = 0; i < param.data.length; i++) param.data[i]! -= lr * g.data[i]!;
    }
  }

  async predict(features: Float32Array): Promise<{ answer: number; probs: Float32Array }> {
    const { logits } = await this.forward(tensor([1, INPUT_DIM], features));
    const probs = softmaxRows(logits.data, 1, NUM_CLASSES);
    let best = 0;
    for (let i = 1; i < NUM_CLASSES; i++) if (probs[i]! > probs[best]!) best = i;
    return { answer: best, probs };
  }
}
