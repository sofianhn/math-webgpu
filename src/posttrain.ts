/**
 * Post-training hooks (stubs) — what can improve a digit-class MLP
 * without English chain-of-thought.
 *
 * The model has no language channel: it maps place-value features → digit
 * heads. Useful post-train signals stay numeric:
 *
 * 1. Digit RL / correctness reward — sample answers, reward exact match
 *    (or digit-wise partial credit on H/T/O).
 * 2. DPO-style preference on answer digits — preferred vs rejected integer
 *    (or digit triple) for the same expression.
 * 3. Self-consistency — multiple stochastic forward passes / temperature;
 *    keep the mode (or majority vote over digit heads).
 *
 * Verbal scratchpads / English CoT cannot be learned here — there is no
 * token sequence to emit reasoning into.
 */
import type { Model, TrainingStage } from "./types";

export type PostTrainKind = "digit-rl" | "dpo-digits" | "self-consistency";

export interface PostTrainConfig {
  kind: PostTrainKind;
  /** Learning rate for the post-train stage (when implemented). */
  learningRate?: number;
  /** Max preference / reward steps. */
  steps?: number;
}

/**
 * Stub entry point. Call sites can wire UI “Post-train (soon)” without
 * inventing a new stage string — maps onto existing TrainingStage "rl"
 * or "finetune" when implemented.
 */
export function describePostTrain(kind: PostTrainKind): string {
  switch (kind) {
    case "digit-rl":
      return "Reward exact (or digit-wise) correctness — no English needed.";
    case "dpo-digits":
      return "Prefer correct answer digits over near-misses for the same expression.";
    case "self-consistency":
      return "Majority-vote digit heads / temperature samples at inference.";
  }
}

export function postTrainStage(kind: PostTrainKind): TrainingStage {
  if (kind === "self-consistency") return "pretrain"; // inference-only today
  return "rl";
}

/**
 * Placeholder — throws until a StageStrategy + reward/preference pipeline lands.
 * Kept so the UI / blog can link a real function name.
 */
export async function runPostTrain(
  _model: Model,
  config: PostTrainConfig,
): Promise<never> {
  throw new Error(
    `post-train "${config.kind}" not implemented yet — ${describePostTrain(config.kind)}`,
  );
}
