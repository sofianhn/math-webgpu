import { CpuBackend } from "../src/webgpu";
import { TinyMathMlp } from "../src/model";
import {
  ArithmeticDataset,
  MathDataPipeline,
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

const EASY: Array<readonly [number, Op, number]> = [
  [3, "+", 5],
  [12, "+", 7],
  [9, "-", 4],
  [20, "-", 8],
  [6, "*", 7],
  [48, "/", 6],
];
const HARD: Array<readonly [number, Op, number]> = [
  [47, "+", 38],
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

async function probe(model: TinyMathMlp, probes: Array<readonly [number, Op, number]>) {
  let ok = 0;
  for (const [a, op, b] of probes) {
    const t = computeAnswer(a, op, b);
    if (t >= 1000) continue;
    if ((await model.predict(encodeFeatures(a, op, b))).answer === t) ok++;
  }
  return ok / probes.length;
}

async function runOnce(cfg: {
  name: string;
  hidden: number;
  epochs: number;
  lr: number;
  examples: number;
  batch: number;
}) {
  const model = new TinyMathMlp(new CpuBackend(), cfg.hidden);
  const filter = (a: number, b: number, op: Op) => !isExpressionHeldOut(a, op, b);
  const pipe = new MathDataPipeline("pretrain", new ArithmeticDataset(filter));
  const held = batchFromExprs(buildExpressionHoldout(160));
  const steps = Math.max(1, Math.round(cfg.examples / cfg.batch));
  await new MathTrainer().train(model, pipe, {
    stage: "pretrain",
    epochs: cfg.epochs,
    batchSize: cfg.batch,
    learningRate: cfg.lr,
    stepsPerEpoch: steps,
    heldoutBatch: held,
  });
  const hold = (await evaluateModel(model, held)).accuracy;
  const train = (await new MathTrainer().evaluate(model, pipe, 12)).accuracy;
  const easy = await probe(model, EASY);
  const hard = await probe(model, HARD);
  return { train, hold, easy, hard };
}

async function multi(cfg: {
  name: string;
  hidden: number;
  epochs: number;
  lr: number;
  examples: number;
  batch: number;
}, seeds = 3) {
  const rows = [];
  for (let i = 0; i < seeds; i++) {
    const r = await runOnce(cfg);
    rows.push(r);
    console.log(
      `  seed${i + 1}`,
      `train ${(r.train * 100).toFixed(1)}% hold ${(r.hold * 100).toFixed(1)}% easy ${(r.easy * 100).toFixed(0)}% hard ${(r.hard * 100).toFixed(0)}%`,
    );
  }
  const avg = (k: "train" | "hold" | "easy" | "hard") =>
    rows.reduce((s, r) => s + r[k], 0) / rows.length;
  console.log(
    `=> ${cfg.name} AVG train ${(avg("train") * 100).toFixed(1)}% hold ${(avg("hold") * 100).toFixed(1)}% easy ${(avg("easy") * 100).toFixed(0)}% hard ${(avg("hard") * 100).toFixed(0)}%`,
  );
}

async function main() {
  const picks = [
    { name: "SIMPLE", hidden: 96, epochs: 50, lr: 0.012, examples: 2560, batch: 64 },
    { name: "SIMPLE+", hidden: 112, epochs: 55, lr: 0.01, examples: 3072, batch: 64 },
    { name: "NINJA", hidden: 256, epochs: 120, lr: 0.006, examples: 8192, batch: 64 },
    { name: "NINJA+", hidden: 256, epochs: 150, lr: 0.005, examples: 10240, batch: 64 },
  ];
  for (const p of picks) {
    console.log("confirm", p.name, p);
    await multi(p, 3);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
