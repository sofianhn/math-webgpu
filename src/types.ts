/** Training stages the architecture is designed to support. */
export type TrainingStage = "pretrain" | "finetune" | "rl";

/** Dense float32 tensor living in CPU memory (GPU buffers wrap these via WebGpuBackend). */
export interface Tensor {
  shape: readonly number[];
  data: Float32Array;
}

export interface Batch {
  /** Flattened [batch, inputDim] place-value features. */
  inputs: Tensor;
  /**
   * Digit targets [batch * answerDigits], LSD-first, each entry 0–9.
   * Zero-padded to ANSWER_DIGITS places (leading high places are 0).
   */
  digitTargets: Uint8Array;
  /**
   * Per-example count of digit places to supervise (length = batch).
   * Includes trailing zeros out to the curriculum frontier so high places
   * learn to stay 0 — otherwise a shared digit head copies the ones digit upward.
   */
  activeCounts?: Uint8Array;
  /** Exact answers as decimal strings (BigInt-safe). */
  answers?: string[];
  /** Optional human-readable expressions for logging. */
  texts?: string[];
}

export interface Dataset {
  readonly name: string;
  size(): number;
  /** Draw a batch; used by all stages (pretrain / finetune / rl). */
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
  /**
   * Digit logits flattened [batch, answerDigits * 10].
   * Place p occupies columns [p*10, (p+1)*10).
   */
  logits: Tensor;
  /** Intermediate activations needed for CPU backward (hybrid path). */
  cache: Record<string, Tensor>;
}

export interface Model {
  readonly name: string;
  readonly inputDim: number;
  /** Number of answer digit places predicted. */
  readonly answerDigits: number;
  parameters(): { name: string; tensor: Tensor }[];
  forward(inputs: Tensor): Promise<ForwardResult>;
  /** Hybrid: gradients computed on CPU from cached activations + digit targets. */
  backward(
    cache: Record<string, Tensor>,
    digitTargets: Uint8Array,
    activeCounts?: Uint8Array,
  ): Map<string, Tensor>;
  applyGradients(grads: Map<string, Tensor>, lr: number): void;
}

export interface EvalSliceMetrics {
  loss: number;
  perplexity: number;
  /** Exact-match accuracy (all digits correct). */
  accuracy: number;
  /** Mean per-digit accuracy. */
  digitAccuracy: number;
  n: number;
  tokens: number;
}

export interface TrainMetrics {
  epoch: number;
  loss: number;
  /** Train perplexity = exp(mean digit CE). */
  perplexity: number;
  accuracy: number;
  digitAccuracy: number;
  stage: TrainingStage;
  /** Examples seen this epoch (steps × batch). */
  examples: number;
  /** Tokens seen this epoch (examples × tokens/example). */
  tokens: number;
  /** Current curriculum max operand digits (1–9). */
  maxDigits?: number;
  /** Held-out eval the model never trained on (optional). */
  heldout?: EvalSliceMetrics;
}

export interface StageStrategy {
  /** Optional per-stage loss / reward shaping. Default: digit CE for pretrain. */
  loss?(logits: Tensor, digitTargets: Uint8Array, batchSize: number): number;
  onEpochEnd?(metrics: TrainMetrics): void;
}

export interface TrainerConfig {
  stage: TrainingStage;
  epochs: number;
  batchSize: number;
  learningRate: number;
  /** Gradient steps per epoch (examples/epoch ≈ steps × batch). Default 40. */
  stepsPerEpoch?: number;
  /** Tokens counted per training example (for UI accounting). */
  tokensPerExample?: number;
  /** Optional held-out batch evaluated every epoch (never used for gradients). */
  heldoutBatch?: Batch;
  /**
   * Optional held-out expressions; when set with curriculum, each epoch evaluates
   * only exprs whose operands fit the current maxDigits (plus full bank at end).
   */
  heldoutExprs?: ReadonlyArray<readonly [bigint, string, bigint]>;
  strategy?: StageStrategy;
  /** Called after each epoch; may be async so the UI can paint mid-train. */
  onMetrics?: (m: TrainMetrics) => void | Promise<void>;
  /**
   * Curriculum: max operand digits at epoch 1 and at final epoch.
   * Linearly interpolates (rounded) when both set.
   */
  curriculumStartDigits?: number;
  curriculumEndDigits?: number;
  /** Hook to push the active max-digit length into the dataset each epoch. */
  onCurriculumDigits?: (maxDigits: number) => void;
}

export interface Trainer {
  train(model: Model, pipeline: DataPipeline, config: TrainerConfig): Promise<TrainMetrics[]>;
  evaluate(
    model: Model,
    pipeline: DataPipeline,
    batches?: number,
  ): Promise<{ loss: number; accuracy: number; digitAccuracy: number; perplexity: number }>;
}
