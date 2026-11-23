import type { ForwardResult, Model, Tensor } from "./types";
import { WebGpuBackend } from "./webgpu";
import { addBiasInPlace, randn, reluInPlace, softmaxRows, tensor, zeros } from "./tensor";
import { INPUT_DIM, NUM_CLASSES } from "./data";

export const DEFAULT_HIDDEN = 96;

/**
 * Tiny 2-layer MLP: input → ReLU(W1x+b1) → W2h+b2 → logits.
 * Forward GEMMs run on WebGPU; ReLU/bias/softmax/backward stay on CPU (hybrid).
 */
export class TinyMathMlp implements Model {
  readonly name = "tiny-math-mlp";
  readonly inputDim = INPUT_DIM;
  readonly numClasses = NUM_CLASSES;
  readonly hidden: number;

  private readonly gpu: WebGpuBackend;
  private readonly W1: Tensor;
  private readonly b1: Tensor;
  private readonly W2: Tensor;
  private readonly b2: Tensor;

  constructor(gpu: WebGpuBackend, hidden = DEFAULT_HIDDEN) {
    this.gpu = gpu;
    this.hidden = hidden;
    // Xavier-ish scales
    this.W1 = randn([INPUT_DIM, hidden], Math.sqrt(2 / INPUT_DIM));
    this.b1 = zeros([hidden]);
    this.W2 = randn([hidden, NUM_CLASSES], Math.sqrt(2 / hidden));
    this.b2 = zeros([NUM_CLASSES]);
  }

  /** L2 norms per parameter tensor — useful for “how much did weights move?” viz. */
  weightNorms(): { name: string; l2: number; count: number }[] {
    return this.parameters().map(({ name, tensor: t }) => {
      let s = 0;
      for (let i = 0; i < t.data.length; i++) s += t.data[i] * t.data[i];
      return { name, l2: Math.sqrt(s), count: t.data.length };
    });
  }

  parameters() {
    return [
      { name: "W1", tensor: this.W1 },
      { name: "b1", tensor: this.b1 },
      { name: "W2", tensor: this.W2 },
      { name: "b2", tensor: this.b2 },
    ];
  }

  async forward(inputs: Tensor): Promise<ForwardResult> {
    const [B, D] = inputs.shape;
    const H = this.hidden;
    if (D !== INPUT_DIM) throw new Error(`expected inputDim ${INPUT_DIM}, got ${D}`);

    // h_pre = X @ W1
    const hPreData = await this.gpu.matmul(inputs.data, this.W1.data, B, INPUT_DIM, H);
    addBiasInPlace(hPreData, B, H, this.b1.data);
    const hData = hPreData.slice();
    reluInPlace(hData);

    // logits = H @ W2
    const logitsData = await this.gpu.matmul(hData, this.W2.data, B, H, NUM_CLASSES);
    addBiasInPlace(logitsData, B, NUM_CLASSES, this.b2.data);

    return {
      logits: tensor([B, NUM_CLASSES], logitsData),
      cache: {
        x: inputs,
        hPre: tensor([B, H], hPreData),
        h: tensor([B, H], hData),
        logits: tensor([B, NUM_CLASSES], logitsData),
      },
    };
  }

  backward(cache: Record<string, Tensor>, targets: Int32Array): Map<string, Tensor> {
    const x = cache.x;
    const h = cache.h;
    const hPre = cache.hPre;
    const logits = cache.logits;
    const [B] = x.shape;
    const H = this.hidden;

    const probs = softmaxRows(logits.data, B, NUM_CLASSES);
    // dL/dlogits
    const dLogits = probs.slice();
    for (let i = 0; i < B; i++) dLogits[i * NUM_CLASSES + targets[i]] -= 1;
    for (let i = 0; i < dLogits.length; i++) dLogits[i] /= B;

    // dW2 = H^T @ dLogits ; db2 = sum rows
    const dW2 = zeros([H, NUM_CLASSES]);
    const db2 = zeros([NUM_CLASSES]);
    for (let i = 0; i < B; i++) {
      for (let k = 0; k < H; k++) {
        const hk = h.data[i * H + k];
        for (let j = 0; j < NUM_CLASSES; j++) {
          const g = dLogits[i * NUM_CLASSES + j];
          dW2.data[k * NUM_CLASSES + j] += hk * g;
        }
      }
      for (let j = 0; j < NUM_CLASSES; j++) db2.data[j] += dLogits[i * NUM_CLASSES + j];
    }

    // dH = dLogits @ W2^T
    const dH = new Float32Array(B * H);
    for (let i = 0; i < B; i++) {
      for (let k = 0; k < H; k++) {
        let s = 0;
        for (let j = 0; j < NUM_CLASSES; j++) {
          s += dLogits[i * NUM_CLASSES + j] * this.W2.data[k * NUM_CLASSES + j];
        }
        dH[i * H + k] = s;
      }
    }

    // ReLU backward through hPre
    const dHPre = dH.slice();
    for (let i = 0; i < dHPre.length; i++) {
      if (hPre.data[i] <= 0) dHPre[i] = 0;
    }

    const dW1 = zeros([INPUT_DIM, H]);
    const db1 = zeros([H]);
    for (let i = 0; i < B; i++) {
      for (let d = 0; d < INPUT_DIM; d++) {
        const xd = x.data[i * INPUT_DIM + d];
        for (let k = 0; k < H; k++) {
          const g = dHPre[i * H + k];
          dW1.data[d * H + k] += xd * g;
        }
      }
      for (let k = 0; k < H; k++) db1.data[k] += dHPre[i * H + k];
    }

    return new Map([
      ["W1", dW1],
      ["b1", db1],
      ["W2", dW2],
      ["b2", db2],
    ]);
  }

  applyGradients(grads: Map<string, Tensor>, lr: number): void {
    const pairs: [Tensor, Tensor | undefined][] = [
      [this.W1, grads.get("W1")],
      [this.b1, grads.get("b1")],
      [this.W2, grads.get("W2")],
      [this.b2, grads.get("b2")],
    ];
    for (const [param, g] of pairs) {
      if (!g) continue;
      for (let i = 0; i < param.data.length; i++) param.data[i] -= lr * g.data[i];
    }
  }

  /** Sync predict for the demo UI (one sample). */
  async predict(features: Float32Array): Promise<{ answer: number; probs: Float32Array }> {
    const { logits } = await this.forward(tensor([1, INPUT_DIM], features));
    const probs = softmaxRows(logits.data, 1, NUM_CLASSES);
    let best = 0;
    for (let i = 1; i < NUM_CLASSES; i++) if (probs[i] > probs[best]) best = i;
    return { answer: best, probs };
  }
}
