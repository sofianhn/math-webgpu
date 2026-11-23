import type { Tensor } from "./types";

export function tensor(shape: number[], data?: Float32Array | number[]): Tensor {
  const size = shape.reduce((a, b) => a * b, 1);
  const arr =
    data instanceof Float32Array
      ? data
      : data
        ? Float32Array.from(data)
        : new Float32Array(size);
  if (arr.length !== size) throw new Error(`tensor length ${arr.length} != ${size}`);
  return { shape, data: arr };
}

export function zeros(shape: number[]): Tensor {
  return tensor(shape);
}

export function randn(shape: number[], scale = 0.1): Tensor {
  const t = zeros(shape);
  for (let i = 0; i < t.data.length; i++) {
    // Box-Muller
    const u = 1 - Math.random();
    const v = Math.random();
    t.data[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * scale;
  }
  return t;
}

export function addBiasInPlace(mat: Float32Array, rows: number, cols: number, bias: Float32Array): void {
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      mat[r * cols + c] += bias[c];
    }
  }
}

export function reluInPlace(x: Float32Array): void {
  for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0;
}

export function softmaxRows(logits: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(logits.length);
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    let max = -Infinity;
    for (let c = 0; c < cols; c++) max = Math.max(max, logits[off + c]);
    let sum = 0;
    for (let c = 0; c < cols; c++) {
      const e = Math.exp(logits[off + c] - max);
      out[off + c] = e;
      sum += e;
    }
    for (let c = 0; c < cols; c++) out[off + c] /= sum;
  }
  return out;
}

/** Cross-entropy over rows of logits vs integer targets. */
export function crossEntropy(logits: Float32Array, rows: number, cols: number, targets: Int32Array): number {
  let loss = 0;
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    let max = -Infinity;
    for (let c = 0; c < cols; c++) max = Math.max(max, logits[off + c]);
    let sum = 0;
    for (let c = 0; c < cols; c++) sum += Math.exp(logits[off + c] - max);
    const t = targets[r];
    loss += -(logits[off + t] - max - Math.log(sum));
  }
  return loss / rows;
}

export function argmaxRow(probs: Float32Array, rows: number, cols: number): Int32Array {
  const out = new Int32Array(rows);
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    let best = 0;
    let bestV = probs[off];
    for (let c = 1; c < cols; c++) {
      if (probs[off + c] > bestV) {
        bestV = probs[off + c];
        best = c;
      }
    }
    out[r] = best;
  }
  return out;
}
