/**
 * Finish Ninja by continuing Mid (7-digit) → 9-digit curriculum + Digit RL.
 * Faster than training Ninja from scratch after a hung long run.
 */
import { mkdirSync, writeFileSync, readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { CpuBackend } from "../src/webgpu";
import { TinyMathMlp } from "../src/model";
import {
  ANSWER_DIGITS,
  ArithmeticDataset,
  CURRICULUM,
  MathDataPipeline,
  NINJA_MIX,
  OPERAND_DIGITS,
  computeAnswer,
  encodeFeatures,
  type Op,
} from "../src/data";
import { MathTrainer } from "../src/trainer";
import {
  FAIR_HOLDOUT_PCT,
  FAIR_HOLDOUT_SEED,
  batchFromExprs,
  buildExpressionHoldout,
  buildLengthProbes,
  evaluateModel,
  isExpressionHeldOut,
} from "../src/evals";
import {
  CHECKPOINT_VERSION,
  buildCheckpointMeta,
  loadWeights,
  packCheckpoint,
  type CheckpointFile,
  type CheckpointManifest,
} from "../src/checkpoint";
import { runDigitRl } from "../src/posttrain";
import { MODE_PRESETS } from "../src/presets";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT_DIR = join(ROOT, "public", "checkpoints");
const epochs = Number(process.env.EPOCHS ?? 90);
const examples = Number(process.env.EXAMPLES ?? 12288);
const batch = 64;
const lr = Number(process.env.LR ?? 0.0025);
const rlSteps = Number(process.env.RL_STEPS ?? 80);

async function probeAcc(
  model: TinyMathMlp,
  probes: Array<readonly [bigint, Op, bigint]>,
): Promise<number> {
  let ok = 0;
  for (const [a, op, b] of probes) {
    const truth = computeAnswer(a, op, b);
    const { answer } = await model.predict(encodeFeatures(a, op, b));
    if (answer === truth) ok++;
  }
  return probes.length === 0 ? 0 : ok / probes.length;
}

async function main() {
  const midPath = join(OUT_DIR, "mid.json");
  if (!existsSync(midPath)) throw new Error(`missing ${midPath}`);
  const mid = JSON.parse(readFileSync(midPath, "utf8")) as CheckpointFile;

  const hidden = mid.hyperparams.hidden; // keep Mid capacity (384) — still 9-digit capable
  const gpu = new CpuBackend();
  const model = new TinyMathMlp(gpu, hidden);
  loadWeights(model, mid.weights.params);

  const filter = (a: bigint, b: bigint, op: Op) => !isExpressionHeldOut(a, op, b);
  const dataset = new ArithmeticDataset({
    filter,
    mix: NINJA_MIX,
    hardBias: 0.55,
    maxDigits: 7,
  });
  const pipe = new MathDataPipeline("pretrain", dataset);
  const holdExprs = buildExpressionHoldout(256, FAIR_HOLDOUT_PCT, OPERAND_DIGITS);
  const heldBatch = batchFromExprs(holdExprs);
  const steps = Math.max(1, Math.round(examples / batch));
  const trainer = new MathTrainer();

  console.log(
    `\n=== Ninja finetune from Mid · H=${hidden} · ep=${epochs} · curr 7→9 · steps=${steps} ===`,
  );
  const before = await evaluateModel(model, heldBatch);
  console.log(
    `  start hold exact=${(before.accuracy * 100).toFixed(1)}% dig=${(before.digitAccuracy * 100).toFixed(1)}%`,
  );

  const t0 = Date.now();
  let lastD = 7;
  const hist = await trainer.train(model, pipe, {
    stage: "pretrain",
    epochs,
    batchSize: batch,
    learningRate: lr,
    stepsPerEpoch: steps,
    heldoutBatch: heldBatch,
    heldoutExprs: holdExprs,
    curriculumStartDigits: 7,
    curriculumEndDigits: 9,
    onCurriculumDigits: (d) => {
      if (d !== lastD) {
        console.log(`  ↳ curriculum → maxDigits=${d}`);
        lastD = d;
      }
      dataset.setMaxDigits(d);
    },
    onMetrics: (m) => {
      if (m.epoch === 1 || m.epoch === epochs || m.epoch % 10 === 0) {
        const hold = m.heldout
          ? ` holdExact=${(m.heldout.accuracy * 100).toFixed(1)}% dig=${(m.heldout.digitAccuracy * 100).toFixed(1)}%`
          : "";
        console.log(
          `  ep ${String(m.epoch).padStart(3)}  d=${m.maxDigits}  loss=${m.loss.toFixed(4)}  exact=${(m.accuracy * 100).toFixed(1)}%  digit=${(m.digitAccuracy * 100).toFixed(1)}%${hold}`,
        );
      }
    },
  });

  let finalHoldout = await evaluateModel(model, heldBatch);
  console.log(
    `  pretrain done in ${((Date.now() - t0) / 1000).toFixed(0)}s · holdExact=${(finalHoldout.accuracy * 100).toFixed(1)}%`,
  );

  console.log(`\n=== Digit RL (${rlSteps} steps) ===`);
  const rl = await runDigitRl(model, {
    steps: rlSteps,
    learningRate: 0.0007,
    holdoutExprs: holdExprs,
    maxDigits: OPERAND_DIGITS,
    onStep: (m) => {
      if (m.step === m.steps || m.step % Math.max(1, Math.floor(rlSteps / 4)) === 0) {
        console.log(
          `  rl ${m.step}/${m.steps} reward=${m.meanReward.toFixed(3)} exact=${(m.exactRate * 100).toFixed(1)}%`,
        );
      }
    },
  });
  finalHoldout = rl.afterHoldout;
  console.log(
    `  RL hold ${(rl.beforeHoldout.accuracy * 100).toFixed(1)}% → ${(rl.afterHoldout.accuracy * 100).toFixed(1)}% (${(rl.holdoutDelta * 100).toFixed(1)} pp)`,
  );

  const byLength: Record<string, number> = {};
  for (const bucket of buildLengthProbes(9, 24)) {
    byLength[String(bucket.digits)] = await probeAcc(model, bucket.exprs);
  }
  const curriculum = await probeAcc(model, [...CURRICULUM]);
  console.log(
    `  curr=${(curriculum * 100).toFixed(0)}% · byLength ${Object.entries(byLength)
      .map(([d, a]) => `${d}=${(a * 100).toFixed(0)}%`)
      .join(" · ")}`,
  );

  const preset = MODE_PRESETS.ninja;
  const meta = buildCheckpointMeta({
    id: "ninja",
    label: "Ninja",
    blurb: `Nine-digit calculator · continued from Mid → 9 digits + Digit RL. Hold-out ${(rl.beforeHoldout.accuracy * 100).toFixed(1)}% → ${(finalHoldout.accuracy * 100).toFixed(1)}%.`,
    hyperparams: {
      hidden,
      epochs: mid.hyperparams.epochs + epochs,
      lr,
      examples,
      batch,
      mix: "ninja",
      hardBias: 0.55,
      heldout: true,
      curriculumStart: 1,
      curriculumEnd: 9,
      operandDigits: OPERAND_DIGITS,
      answerDigits: ANSWER_DIGITS,
    },
    holdout: { pct: FAIR_HOLDOUT_PCT, n: holdExprs.length, seed: FAIR_HOLDOUT_SEED },
    curves: [...(mid.curves ?? []), ...hist],
    finalHoldout,
    probes: {
      easy: mid.probes?.easy ?? 0,
      hard: mid.probes?.hard ?? 0,
      curriculum,
      byLength,
    },
    paramCount: TinyMathMlp.paramCount(hidden),
  });
  const ckpt = packCheckpoint(meta, model);
  ckpt.postTrain = {
    kind: "digit-rl",
    steps: rl.steps,
    meanReward: rl.meanReward,
    beforeHoldout: rl.beforeHoldout,
    afterHoldout: rl.afterHoldout,
    holdoutDelta: rl.holdoutDelta,
    baseId: "mid",
  };
  mkdirSync(OUT_DIR, { recursive: true });
  writeFileSync(join(OUT_DIR, "ninja.json"), JSON.stringify(ckpt));
  writeFileSync(join(OUT_DIR, "ninja-digit-rl.json"), JSON.stringify(ckpt));

  // rebuild manifest
  const models: CheckpointManifest["models"] = [];
  for (const id of ["simple", "mid", "ninja"] as const) {
    const p = join(OUT_DIR, `${id}.json`);
    if (!existsSync(p)) continue;
    const c = JSON.parse(readFileSync(p, "utf8")) as CheckpointFile;
    models.push({
      id,
      label: c.label,
      blurb: c.blurb,
      path: `checkpoints/${id}.json`,
      hidden: c.hyperparams.hidden,
      epochs: c.hyperparams.epochs,
      paramCount: c.arch.paramCount,
      finalHoldoutAcc: c.finalHoldout.accuracy,
      curriculumEnd: c.hyperparams.curriculumEnd,
    });
  }
  writeFileSync(
    join(OUT_DIR, "manifest.json"),
    JSON.stringify({ version: CHECKPOINT_VERSION, models }, null, 2) + "\n",
  );
  console.log(`\nWrote ninja.json · hold ${(finalHoldout.accuracy * 100).toFixed(1)}%`);
  console.log(`Manifest → ${join(OUT_DIR, "manifest.json")}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
