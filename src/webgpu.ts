/**
 * Minimal WebGPU backend: device init + matmul compute shader.
 *
 * Extension path:
 * - Add more kernels (bias-add, relu, softmax, gemm-batched).
 * - Upload weights once; keep activations on-GPU across layers.
 * - Move backward into WGSL (dW = X^T @ dY, etc.) for full GPU training.
 * Today: GPU matmul for forward GEMMs; caller copies results back for CPU loss/backward.
 */

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

export class WebGpuBackend {
  readonly device: GPUDevice;
  private readonly matmulPipeline: GPUComputePipeline;
  private readonly dimsBuffer: GPUBuffer;

  private constructor(device: GPUDevice, pipeline: GPUComputePipeline, dimsBuffer: GPUBuffer) {
    this.device = device;
    this.matmulPipeline = pipeline;
    this.dimsBuffer = dimsBuffer;
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
    const dimsBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    return new WebGpuBackend(device, pipeline, dimsBuffer);
  }

  /** C[M,N] = A[M,K] @ B[K,N] on GPU; returns Float32Array of length M*N. */
  async matmul(A: Float32Array, B: Float32Array, M: number, K: number, N: number): Promise<Float32Array> {
    if (A.length !== M * K) throw new Error(`A size ${A.length} != ${M}*${K}`);
    if (B.length !== K * N) throw new Error(`B size ${B.length} != ${K}*${N}`);

    const aBuf = this.upload(A, GPUBufferUsage.STORAGE);
    const bBuf = this.upload(B, GPUBufferUsage.STORAGE);
    const cSize = M * N * 4;
    const cBuf = this.device.createBuffer({
      size: Math.max(4, cSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const dims = new Uint32Array([M, K, N, 0]);
    this.device.queue.writeBuffer(this.dimsBuffer, 0, dims);

    const bindGroup = this.device.createBindGroup({
      layout: this.matmulPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: bBuf } },
        { binding: 2, resource: { buffer: cBuf } },
        { binding: 3, resource: { buffer: this.dimsBuffer } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.matmulPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(M / 8), Math.ceil(N / 8));
    pass.end();

    const readBuf = this.device.createBuffer({
      size: Math.max(4, cSize),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(cBuf, 0, readBuf, 0, cSize);
    this.device.queue.submit([encoder.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();

    aBuf.destroy();
    bBuf.destroy();
    cBuf.destroy();
    readBuf.destroy();

    return copy.slice(0, M * N);
  }

  private upload(data: Float32Array, usage: GPUBufferUsageFlags): GPUBuffer {
    const buf = this.device.createBuffer({
      size: Math.max(4, data.byteLength),
      usage: usage | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
  }
}
