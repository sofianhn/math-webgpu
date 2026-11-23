/**
 * Empirical Simple vs Ninja preset search (CPU).
 * Run: npx tsx scripts/sweep-presets.ts
 */
import { CpuBackend } from "../src/webgpu";
import { TinyMathMlp } from "../src/model";
import {
  ArithmeticDataset,
  MathDataPipeline,
  OPS,
  computeAnswer,
  encodeFeatures,
  type Op,
} from "../src/data";
import { MathTrainer } from "../src/trainer";
import {
  batchFromExprs,
  buildExpressionHoldout,
  isExpressionHeldOut,
  evaluateModel,
} from "../src/evals";

type Cfg = {
  name: string;
  hidden: number;
  epochs: number;
  lr: number;
  examples: number;
  batch: number;
};

const EASY: Array<readonly [number, Op, number]> = [
  [3, "+", 5],
  [12, "+", 7],
  [9, "-", 4],
  [20, "-", 8],
  [6, "*", 7],
  [48, "/", 6],
];

const HARD: Array<readonly [number, Op, number]> = [
  [47, "+", 38], // carry
  [83, "-", 29],
  [17, "*", 19],
  [96, "/", 8],
  [58, "+", 67],
  [74, "*", 6],
  [91, "-", 47],
  [36, "*", 12],
  [99, "+", 99],
  [84, "/", 7],
  [25, "*", 25],
  [63, "+", 29],
];

async function probeAcc(model: TinyMathMlp, probes: Array<readonly [number, Op, number]>): Promise<number> {
  let ok = 0;
  for (const [a, op, b] of probes) {
    const truth = computeAnswer(a, op, b);
    if (truth >= 1000) continue;
    const { answer } = await model.predict(encodeFeatures(a, op, b));
    if (answer === truth) ok++;
  }
  return ok / probes.length;
}

async function runCfg(cfg: Cfg) {
  const gpu = new CpuBackend();
  const model = new TinyMathMlp(gpu, cfg.hidden);
  const filter = (a: number, b: number, op: Op) => !isExpressionHeldOut(a, op, b);
  const pipe = new MathDataPipeline("pretrain", new ArithmeticDataset(filter));
  const hold = buildExpressionHoldout(160);
  const heldBatch = batchFromExprs(hold);
  const steps = Math.max(1, Math.round(cfg.examples / cfg.batch));
  const trainer = new MathTrainer();
  const t0 = Date.now();
  const hist = await trainer.train(model, pipe, {
    stage: "pretrain",
    epochs: cfg.epochs,
    batchSize: cfg.batch,
    learningRate: cfg.lr,
    stepsPerEpoch: steps,
    heldoutBatch: heldBatch,
  });
  const last = hist.at(-1)!;
  const held = await evaluateModel(model, heldBatch);
  const easy = await probeAcc(model, EASY);
  const hard = await probeAcc(model, HARD);
  // Random train-dist sample accuracy
  const trainEval = await trainer.evaluate(model, pipe, 10);
  return {
    ...cfg,
    sec: ((Date.now() - t0) / 1000).toFixed(1),
    trainAcc: trainEval.accuracy,
    holdAcc: held.accuracy,
    easy,
    hard,
    finalTrainAcc: last.accuracy,
    holdPpl: held.perplexity,
  };
}

const CANDIDATES: Cfg[] = [
  // Simple candidates — fast, still usable
  { name: "S1", hidden: 48, epochs: 25, lr: 0.02, examples: 1280, batch: 64 },
  { name: "S2", hidden: 64, epochs: 30, lr: 0.015, examples: 1920, batch: 64 },
  { name: "S3", hidden: 64, epochs: 40, lr: 0.01, examples: 2560, batch: 64 },
  { name: "S4", hidden: 80, epochs: 35, lr: 0.012, examples: 2048, batch: 64 },
  // Ninja candidates — heavier, should crush hard probes
  { name: "N1", hidden: 128, epochs: 60, lr: 0.01, examples: 4096, batch: 64 },
  { name: "N2", hidden: 160, epochs: 80, lr: 0.008, examples: 5120, batch: 64 },
  { name: "N3", hidden: 192, epochs: 100, lr: 0.008, examples: 6144, batch: 64 },
  { name: "N4", hidden: 256, epochs: 80, lr: 0.006, examples: 7680, batch: 64 },
  { name: "N5", hidden: 192, epochs: 120, lr: 0.006, examples: 5120, batch: 64 },
  { name: "N6", hidden: 224, epochs: 100, lr: 0.007, examples: 8192, batch: 64 },
];

async function main() {
  console.log("sweep start", CANDIDATES.length, "configs");
  const rows = [];
  for (const cfg of CANDIDATES) {
    console.log("…", cfg.name, cfg);
    const r = await runCfg(cfg);
    rows.push(r);
    console.log(
      `  train=${(r.trainAcc * 100).toFixed(1)}% hold=${(r.holdAcc * 100).toFixed(1)}% easy=${(r.easy * 100).toFixed(0)}% hard=${(r.hard * 100).toFixed(0)}% ${r.sec}s`,
    );
  }
  rows.sort((a, b) => b.hard * 2 + b.holdAcc - (a.hard * 2 + a.holdAcc));
  console.log("\n=== ranked by hard*2 + hold ===");
  for (const r of rows) {
    console.log(
      `${r.name} H=${r.hidden} ep=${r.epochs} lr=${r.lr} ex=${r.examples} | hold ${(r.holdAcc * 100).toFixed(1)}% easy ${(r.easy * 100).toFixed(0)}% hard ${(r.hard * 100).toFixed(0)}% train ${(r.trainAcc * 100).toFixed(1)}% (${r.sec}s)`,
    );
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
