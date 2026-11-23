/** Training stages the architecture is designed to support. Only pretrain is implemented. */
export type TrainingStage = "pretrain" | "finetune" | "rl";

/** Dense float32 tensor living in CPU memory (GPU buffers wrap these via GpuGpuBackend). */
export interface Tensor {
  shape: readonly number[];
  data: Float32Array;
}

export interface Batch {
  /** Flattened [batch, inputDim] features. */
  inputs: Tensor;
  /** Integer class labels length = batch (answer token / class). */
  targets: Int32Array;
  /** Optional human-readable expressions for logging. */
  texts?: string[];
}

export interface Dataset {
  readonly name: string;
  size(): number;
  /** Draw a batch; used by all stages (pretrain today, finetune/rl later). */
  sample(batchSize: number): Batch;
}

/**
 * Data pipeline hook — today wraps a Dataset; later can add tokenization,
 * preference pairs (RL), or instruction packs (finetune) without changing Trainer.
 */
export interface DataPipeline {
  readonly stage: TrainingStage;
  nextBatch(batchSize: number): Batch;
}

export interface ForwardResult {
  /** Logits [batch, numClasses] on CPU (copied back from GPU). */
  logits: Tensor;
  /** Intermediate activations needed for CPU backward (hybrid path). */
  cache: Record<string, Tensor>;
}

export interface Model {
  readonly name: string;
  readonly inputDim: number;
  readonly numClasses: number;
  parameters(): { name: string; tensor: Tensor }[];
  forward(inputs: Tensor): Promise<ForwardResult>;
  /** Hybrid: gradients computed on CPU from cached activations. */
  backward(cache: Record<string, Tensor>, targets: Int32Array): Map<string, Tensor>;
  applyGradients(grads: Map<string, Tensor>, lr: number): void;
}

export interface TrainMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
  stage: TrainingStage;
}

export interface StageStrategy {
  /** Optional per-stage loss / reward shaping. Default: CE for pretrain. */
  loss?(logits: Tensor, targets: Int32Array): number;
  onEpochEnd?(metrics: TrainMetrics): void;
}

export interface TrainerConfig {
  stage: TrainingStage;
  epochs: number;
  batchSize: number;
  learningRate: number;
  strategy?: StageStrategy;
  onMetrics?: (m: TrainMetrics) => void;
}

export interface Trainer {
  train(model: Model, pipeline: DataPipeline, config: TrainerConfig): Promise<TrainMetrics[]>;
  evaluate(model: Model, pipeline: DataPipeline, batches?: number): Promise<{ loss: number; accuracy: number }>;
}
