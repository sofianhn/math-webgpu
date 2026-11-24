/**
 * Harden Ninja on hard multiplies (esp. 47×19).
 * Run: npx tsx scripts/sweep-mul.ts
 */
import { CpuBackend } from "../src/webgpu";
import { TinyMathMlp } from "../src/model";
import {
  ArithmeticDataset,
  MathDataPipeline,
  NINJA_MIX,
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

const HARD_MUL: Array<readonly [number, Op, number]> = [
  [47, "*", 19], // user fail case
  [17, "*", 19],
  [25, "*", 25],
  [36, "*", 12],
  [74, "*", 6],
  [48, "*", 15],
  [23, "*", 27],
  [19, "*", 47],
  [13, "*", 29],
  [47, "+", 38],
  [58, "+", 67],
  [99, "+", 99],
];

async function probe(model: TinyMathMlp, probes: Array<readonly [number, Op, number]>) {
  const rows = [];
  let ok = 0;
  for (const [a, op, b] of probes) {
    const truth = computeAnswer(a, op, b);
    const { answer, probs } = await model.predict(encodeFeatures(a, op, b));
    const hit = answer === truth;
    if (hit) ok++;
    const pTruth = probs[truth] ?? 0;
    const pPred = probs[answer] ?? 0;
    rows.push({
      expr: `${a}${op}${b}`,
      pred: answer,
      truth,
      hit,
      pTruth: +pTruth.toFixed(3),
      pPred: +pPred.toFixed(3),
    });
  }
  return { acc: ok / probes.length, rows };
}

async function run(cfg: {
  name: string;
  hidden: number;
  epochs: number;
  lr: number;
  examples: number;
  batch: number;
  hardBias: number;
}) {
  const model = new TinyMathMlp(new CpuBackend(), cfg.hidden);
  const filter = (a: number, b: number, op: Op) => !isExpressionHeldOut(a, op, b);
  const pipe = new MathDataPipeline(
    "pretrain",
    new ArithmeticDataset({ filter, mix: NINJA_MIX, hardBias: cfg.hardBias }),
  );
  const held = batchFromExprs(buildExpressionHoldout(160));
  const steps = Math.max(1, Math.round(cfg.examples / cfg.batch));
  const t0 = Date.now();
  await new MathTrainer().train(model, pipe, {
    stage: "pretrain",
    epochs: cfg.epochs,
    batchSize: cfg.batch,
    learningRate: cfg.lr,
    stepsPerEpoch: steps,
    heldoutBatch: held,
  });
  const hold = (await evaluateModel(model, held)).accuracy;
  const train = (await new MathTrainer().evaluate(model, pipe, 10)).accuracy;
  const { acc, rows } = await probe(model, HARD_MUL);
  const target = rows.find((r) => r.expr === "47*19");
  console.log(
    cfg.name,
    `H=${cfg.hidden} ep=${cfg.epochs} lr=${cfg.lr} ex=${cfg.examples} hardBias=${cfg.hardBias}`,
    `train ${(train * 100).toFixed(1)}% hold ${(hold * 100).toFixed(1)}% hardMul ${(acc * 100).toFixed(0)}%`,
    `47×19 → ${target?.pred} (want ${target?.truth}) ${target?.hit ? "✓" : "✗"} p(truth)=${target?.pTruth}`,
    `${((Date.now() - t0) / 1000).toFixed(0)}s`,
  );
  return { cfg, train, hold, acc, target };
}

async function main() {
  const cfgs = [
    { name: "A", hidden: 256, epochs: 150, lr: 0.005, examples: 10240, batch: 64, hardBias: 0.5 },
    { name: "B", hidden: 288, epochs: 160, lr: 0.0045, examples: 12288, batch: 64, hardBias: 0.55 },
    { name: "C", hidden: 320, epochs: 160, lr: 0.004, examples: 12288, batch: 64, hardBias: 0.55 },
    { name: "D", hidden: 288, epochs: 180, lr: 0.004, examples: 14336, batch: 64, hardBias: 0.6 },
  ];
  for (const c of cfgs) await run(c);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
