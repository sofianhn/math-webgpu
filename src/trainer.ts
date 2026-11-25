import type {
  DataPipeline,
  Model,
  TrainMetrics,
  Trainer,
  TrainerConfig,
  TrainingStage,
} from "./types";
import { digitAccuracy, digitCrossEntropy } from "./tensor";
import { batchFromExprs, evaluateModel, type HoldoutExpr } from "./evals";
import { OPERAND_DIGITS, TOKENS_PER_EXAMPLE, digitLength } from "./data";

const DEFAULT_STAGES: Record<TrainingStage, true> = {
  pretrain: true,
  finetune: true,
  rl: true,
};
void DEFAULT_STAGES;

function curriculumDigits(epoch: number, epochs: number, start: number, end: number): number {
  if (epochs <= 1) return end;
  const t = (epoch - 1) / (epochs - 1);
  return Math.round(start + t * (end - start));
}

/**
 * Training loop with a stage hook. Pretrain uses digit CE;
 * finetune/rl plug in via TrainerConfig.stage + StageStrategy + DataPipeline / posttrain.
 */
export class MathTrainer implements Trainer {
  async train(model: Model, pipeline: DataPipeline, config: TrainerConfig): Promise<TrainMetrics[]> {
    if (pipeline.stage !== config.stage) {
      throw new Error(`pipeline stage ${pipeline.stage} != config stage ${config.stage}`);
    }
    if (config.stage !== "pretrain") {
      throw new Error(
        `stage "${config.stage}" not implemented in MathTrainer — use posttrain helpers for rl/dpo.`,
      );
    }

    const history: TrainMetrics[] = [];
    const stepsPerEpoch = config.stepsPerEpoch ?? 40;
    const tokensPerExample = config.tokensPerExample ?? TOKENS_PER_EXAMPLE;
    const currStart = config.curriculumStartDigits ?? 1;
    const currEnd = config.curriculumEndDigits ?? OPERAND_DIGITS;
    const m = new Map<string, Float32Array>();
    const v = new Map<string, Float32Array>();
    let t = 0;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;
    const places = model.answerDigits;

    for (let epoch = 1; epoch <= config.epochs; epoch++) {
      const maxDigits = curriculumDigits(epoch, config.epochs, currStart, currEnd);
      config.onCurriculumDigits?.(maxDigits);

      let lossSum = 0;
      let exactSum = 0;
      let digitSum = 0;
      let total = 0;

      for (let step = 0; step < stepsPerEpoch; step++) {
        const batch = pipeline.nextBatch(config.batchSize);
        const { logits, cache } = await model.forward(batch.inputs);
        const [B] = logits.shape;

        const loss = config.strategy?.loss
          ? config.strategy.loss(logits, batch.digitTargets, B)
          : digitCrossEntropy(logits.data, B, places, batch.digitTargets, 10, batch.activeCounts);

        const acc = digitAccuracy(
          logits.data,
          B,
          places,
          batch.digitTargets,
          10,
          batch.activeCounts,
        );
        exactSum += acc.exact * B;
        digitSum += acc.digit * B;
        total += B;
        lossSum += loss;

        const grads = model.backward(cache, batch.digitTargets, batch.activeCounts);
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
        accuracy: exactSum / total,
        digitAccuracy: digitSum / total,
        stage: config.stage,
        examples: total,
        tokens: total * tokensPerExample,
        maxDigits,
      };

      if (config.heldoutExprs && config.heldoutExprs.length > 0) {
        const matched = (config.heldoutExprs as HoldoutExpr[]).filter(
          ([a, , b]) => digitLength(a) <= maxDigits && digitLength(b) <= maxDigits,
        );
        const slice = matched.length > 0 ? matched.slice(0, 160) : (config.heldoutExprs as HoldoutExpr[]).slice(0, 160);
        metrics.heldout = await evaluateModel(model, batchFromExprs(slice));
      } else if (config.heldoutBatch && config.heldoutBatch.digitTargets.length > 0) {
        metrics.heldout = await evaluateModel(model, config.heldoutBatch);
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
  ): Promise<{ loss: number; accuracy: number; digitAccuracy: number; perplexity: number }> {
    const batchSize = 64;
    const places = model.answerDigits;
    let lossSum = 0;
    let exactSum = 0;
    let digitSum = 0;
    let total = 0;

    for (let i = 0; i < batches; i++) {
      const batch = pipeline.nextBatch(batchSize);
      const { logits } = await model.forward(batch.inputs);
      const [B] = logits.shape;
      lossSum += digitCrossEntropy(logits.data, B, places, batch.digitTargets, 10, batch.activeCounts);
      const acc = digitAccuracy(logits.data, B, places, batch.digitTargets, 10, batch.activeCounts);
      exactSum += acc.exact * B;
      digitSum += acc.digit * B;
      total += B;
    }

    const loss = lossSum / batches;
    return {
      loss,
      accuracy: exactSum / total,
      digitAccuracy: digitSum / total,
      perplexity: Math.exp(loss),
    };
  }
}
