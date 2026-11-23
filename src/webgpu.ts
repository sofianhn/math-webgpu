/**
 * Minimal WebGPU backend: device init + matmul compute shader.
 *
 * Extension path:
 * - Add more kernels (bias-add, relu, softmax, gemm-batched).
 * - Upload weights once; keep activations on-GPU across layers.
 * - Move backward into WGSL (dW = X^T @ dY, etc.) for full GPU training.
 * Today: GPU matmul for forward GEMMs; caller copies results back for CPU loss/backward.
 *
 * Mobile / flaky-driver safety: each matmul is serialized, uses a fresh dims uniform,
 * and is checked against a CPU reference — on mismatch we return the CPU result so
 * training still converges (avoids "loss≈0 but probe wrong" from corrupt GEMMs).
 */

import { cpuMatmul, maxAbsDiff } from "./tensor";

const MATMUL_WGSL = /* wgsl */ `
struct Dims {
  M: u32,
  K: u32,
  N: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= dims.M || col >= dims.N) {
    return;
  }
  var acc = 0.0;
  for (var k = 0u; k < dims.K; k = k + 1u) {
    acc = acc + A[row * dims.K + k] * B[k * dims.N + col];
  }
  C[row * dims.N + col] = acc;
}
`;

function padBytes(nbytes: number): number {
  // Storage buffers: multiple of 4; pad to 16 for broader driver friendliness.
  return Math.max(16, Math.ceil(nbytes / 16) * 16);
}

/** Minimal matmul surface used by TinyMathMlp (GPU or CPU). */
export interface MatmulBackend {
  cpuFallbackCount: number;
  matmul(A: Float32Array, B: Float32Array, M: number, K: number, N: number): Promise<Float32Array>;
}

/** CPU-only backend for sweeps / environments without WebGPU. */
export class CpuBackend implements MatmulBackend {
  cpuFallbackCount = 0;
  async matmul(A: Float32Array, B: Float32Array, M: number, K: number, N: number): Promise<Float32Array> {
    return cpuMatmul(A, B, M, K, N);
  }
}

export class WebGpuBackend implements MatmulBackend {
  readonly device: GPUDevice;
  private readonly matmulPipeline: GPUComputePipeline;
  /** Serialize matmuls so concurrent train/predict cannot race the queue. */
  private chain: Promise<unknown> = Promise.resolve();
  /** How many times CPU fallback replaced a bad GPU result. */
  cpuFallbackCount = 0;

  private constructor(device: GPUDevice, pipeline: GPUComputePipeline) {
    this.device = device;
    this.matmulPipeline = pipeline;
  }

  static async create(): Promise<WebGpuBackend> {
    if (!navigator.gpu) {
      throw new Error("WebGPU unavailable: navigator.gpu is missing. Use Chrome/Edge 113+ or Safari 17+ over HTTPS/localhost.");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("WebGPU: no adapter (GPU blocked or unsupported).");
    const device = await adapter.requestDevice();
    const module = device.createShaderModule({ code: MATMUL_WGSL });
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    return new WebGpuBackend(device, pipeline);
  }

  /** C[M,N] = A[M,K] @ B[K,N]; GPU first, CPU if the driver result disagrees. */
  async matmul(A: Float32Array, B: Float32Array, M: number, K: number, N: number): Promise<Float32Array> {
    if (A.length !== M * K) throw new Error(`A size ${A.length} != ${M}*${K}`);
    if (B.length !== K * N) throw new Error(`B size ${B.length} != ${K}*${N}`);

    const cpu = cpuMatmul(A, B, M, K, N);
    const run = this.chain.then(() => this.matmulGpu(A, B, M, K, N));
    this.chain = run.catch(() => undefined);
    try {
      const gpu = await run;
      const diff = maxAbsDiff(gpu, cpu);
      if (!(diff < 1e-2) || gpu.length !== cpu.length) {
        this.cpuFallbackCount += 1;
        return cpu;
      }
      return gpu;
    } catch {
      this.cpuFallbackCount += 1;
      return cpu;
    }
  }

  private async matmulGpu(A: Float32Array, B: Float32Array, M: number, K: number, N: number): Promise<Float32Array> {
    const aBuf = this.upload(A);
    const bBuf = this.upload(B);
    const cBytes = padBytes(M * N * 4);
    const cBuf = this.device.createBuffer({
      size: cBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Fresh dims uniform per dispatch — avoids races if calls ever overlap.
    const dimsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(dimsBuffer, 0, new Uint32Array([M, K, N, 0]));

    const bindGroup = this.device.createBindGroup({
      layout: this.matmulPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: bBuf } },
        { binding: 2, resource: { buffer: cBuf } },
        { binding: 3, resource: { buffer: dimsBuffer } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.matmulPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(M / 8), Math.ceil(N / 8));
    pass.end();

    const readBuf = this.device.createBuffer({
      size: cBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(cBuf, 0, readBuf, 0, M * N * 4);
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    await readBuf.mapAsync(GPUMapMode.READ);
    const mapped = new Float32Array(readBuf.getMappedRange());
    const out = mapped.slice(0, M * N);
    readBuf.unmap();

    aBuf.destroy();
    bBuf.destroy();
    cBuf.destroy();
    dimsBuffer.destroy();
    readBuf.destroy();

    return out;
  }

  private upload(data: Float32Array): GPUBuffer {
    const buf = this.device.createBuffer({
      size: padBytes(data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
  }
}
