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

async function run(cfg: {
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
  const train = (await new MathTrainer().evaluate(model, pipe, 12)).accuracy;
  const easy = await probe(model, EASY);
  const hard = await probe(model, HARD);
  console.log(
    cfg.name,
    `H=${cfg.hidden} ep=${cfg.epochs} lr=${cfg.lr} ex=${cfg.examples}`,
    `train ${(train * 100).toFixed(1)}% hold ${(hold * 100).toFixed(1)}% easy ${(easy * 100).toFixed(0)}% hard ${(hard * 100).toFixed(0)}% ${((Date.now() - t0) / 1000).toFixed(1)}s`,
  );
}

async function main() {
  const cfgs = [
    { name: "S5", hidden: 96, epochs: 50, lr: 0.01, examples: 3072, batch: 64 },
    { name: "S6", hidden: 96, epochs: 45, lr: 0.012, examples: 2560, batch: 64 },
    { name: "S7", hidden: 112, epochs: 55, lr: 0.01, examples: 3584, batch: 64 },
    { name: "S8", hidden: 128, epochs: 40, lr: 0.01, examples: 3072, batch: 64 },
    { name: "N6b", hidden: 224, epochs: 100, lr: 0.007, examples: 8192, batch: 64 },
    { name: "N7", hidden: 256, epochs: 120, lr: 0.006, examples: 8192, batch: 64 },
    { name: "N8", hidden: 224, epochs: 140, lr: 0.006, examples: 9216, batch: 64 },
  ];
  for (const c of cfgs) await run(c);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
