import type {
  DataPipeline,
  Model,
  TrainMetrics,
  Trainer,
  TrainerConfig,
  TrainingStage,
} from "./types";
import { argmaxRow, crossEntropy, softmaxRows } from "./tensor";
import { evaluateModel } from "./evals";
import { TOKENS_PER_EXAMPLE } from "./data";

const DEFAULT_STAGES: Record<TrainingStage, true> = {
  pretrain: true,
  finetune: true,
  rl: true,
};
void DEFAULT_STAGES;

/**
 * Training loop with a stage hook. Only pretrain is exercised today;
 * finetune/rl plug in via TrainerConfig.stage + StageStrategy + DataPipeline.
 */
export class MathTrainer implements Trainer {
  async train(model: Model, pipeline: DataPipeline, config: TrainerConfig): Promise<TrainMetrics[]> {
    if (pipeline.stage !== config.stage) {
      throw new Error(`pipeline stage ${pipeline.stage} != config stage ${config.stage}`);
    }
    if (config.stage !== "pretrain") {
      throw new Error(
        `stage "${config.stage}" not implemented yet — add a StageStrategy and pipeline, then remove this guard.`,
      );
    }

    const history: TrainMetrics[] = [];
    const stepsPerEpoch = config.stepsPerEpoch ?? 40;
    const tokensPerExample = config.tokensPerExample ?? TOKENS_PER_EXAMPLE;
    const m = new Map<string, Float32Array>();
    const v = new Map<string, Float32Array>();
    let t = 0;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;

    for (let epoch = 1; epoch <= config.epochs; epoch++) {
      let lossSum = 0;
      let correct = 0;
      let total = 0;

      for (let step = 0; step < stepsPerEpoch; step++) {
        const batch = pipeline.nextBatch(config.batchSize);
        const { logits, cache } = await model.forward(batch.inputs);
        const [B, C] = logits.shape;

        const loss = config.strategy?.loss
          ? config.strategy.loss(logits, batch.targets)
          : crossEntropy(logits.data, B, C, batch.targets);

        const probs = softmaxRows(logits.data, B, C);
        const preds = argmaxRow(probs, B, C);
        for (let i = 0; i < B; i++) if (preds[i] === batch.targets[i]) correct++;
        total += B;
        lossSum += loss;

        const grads = model.backward(cache, batch.targets);
        t += 1;
        const lrT = (config.learningRate * Math.sqrt(1 - beta2 ** t)) / (1 - beta1 ** t);
        for (const { name, tensor: param } of model.parameters()) {
          const g = grads.get(name);
          if (!g) continue;
          if (!m.has(name)) {
            m.set(name, new Float32Array(param.data.length));
            v.set(name, new Float32Array(param.data.length));
          }
          const mt = m.get(name)!;
          const vt = v.get(name)!;
          for (let i = 0; i < param.data.length; i++) {
            const gi = g.data[i]!;
            mt[i] = beta1 * mt[i]! + (1 - beta1) * gi;
            vt[i] = beta2 * vt[i]! + (1 - beta2) * gi * gi;
            param.data[i]! -= (lrT * mt[i]!) / (Math.sqrt(vt[i]!) + eps);
          }
        }
      }

      const meanLoss = lossSum / stepsPerEpoch;
      const metrics: TrainMetrics = {
        epoch,
        loss: meanLoss,
        perplexity: Math.exp(meanLoss),
        accuracy: correct / total,
        stage: config.stage,
        examples: total,
        tokens: total * tokensPerExample,
      };

      if (config.heldoutBatch && config.heldoutBatch.targets.length > 0) {
        const held = await evaluateModel(model, config.heldoutBatch);
        metrics.heldout = held;
      }

      history.push(metrics);
      config.strategy?.onEpochEnd?.(metrics);
      await config.onMetrics?.(metrics);
    }

    return history;
  }

  async evaluate(
    model: Model,
    pipeline: DataPipeline,
    batches = 10,
  ): Promise<{ loss: number; accuracy: number; perplexity: number }> {
    const batchSize = 64;
    let lossSum = 0;
    let correct = 0;
    let total = 0;

    for (let i = 0; i < batches; i++) {
      const batch = pipeline.nextBatch(batchSize);
      const { logits } = await model.forward(batch.inputs);
      const [B, C] = logits.shape;
      lossSum += crossEntropy(logits.data, B, C, batch.targets);
      const probs = softmaxRows(logits.data, B, C);
      const preds = argmaxRow(probs, B, C);
      for (let j = 0; j < B; j++) if (preds[j] === batch.targets[j]) correct++;
      total += B;
    }

    const loss = lossSum / batches;
    return { loss, accuracy: correct / total, perplexity: Math.exp(loss) };
  }
}
