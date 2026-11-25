/**
 * Offline CPU pretrain → public/checkpoints/{simple,mid,ninja}.json
 *
 *   npx tsx scripts/pretrain-checkpoints.ts
 *   CHECKPOINT_ONLY=simple npx tsx scripts/pretrain-checkpoints.ts
 *   FAST=1 npx tsx scripts/pretrain-checkpoints.ts   # short smoke curricula
 */
import { mkdirSync, writeFileSync, existsSync, readFileSync } from "node:fs";
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
  packCheckpoint,
  type CheckpointId,
  type CheckpointManifest,
} from "../src/checkpoint";
import { MODE_PRESETS, type ModeId } from "../src/presets";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT_DIR = join(ROOT, "public", "checkpoints");
const FAST = process.env.FAST === "1";

const EASY: Array<readonly [bigint, Op, bigint]> = [
  [3n, "+", 5n],
  [12n, "+", 7n],
  [9n, "-", 4n],
  [20n, "-", 8n],
  [6n, "*", 7n],
  [48n, "/", 6n],
];

const HARD: Array<readonly [bigint, Op, bigint]> = [
  [47n, "+", 38n],
  [83n, "-", 29n],
  [17n, "*", 19n],
  [96n, "/", 8n],
  [1234n, "+", 5678n],
  [99999n, "-", 12345n],
  [123456n, "*", 7n],
  [1000000n, "/", 8n],
  [12345678n, "+", 87654321n],
  [111111111n, "*", 2n],
  [999999999n, "-", 1n],
  [314159265n, "+", 271828182n],
];

async function probeAcc(
  model: TinyMathMlp,
  probes: Array<readonly [bigint, Op, bigint]>,
): Promise<number> {
  let ok = 0;
  let n = 0;
  for (const [a, op, b] of probes) {
    const truth = computeAnswer(a, op, b);
    n++;
    const { answer } = await model.predict(encodeFeatures(a, op, b));
    if (answer === truth) ok++;
  }
  return n === 0 ? 0 : ok / n;
}

function fastPreset(id: ModeId) {
  const p = MODE_PRESETS[id];
  if (!FAST) return p;
  return {
    ...p,
    epochs: id === "simple" ? 12 : id === "mid" ? 16 : 20,
    examples: 2048,
    batch: 64,
  };
}

async function trainOne(id: ModeId): Promise<CheckpointManifest["models"][number]> {
  const preset = fastPreset(id);
  const gpu = new CpuBackend();
  const model = new TinyMathMlp(gpu, preset.hidden);
  const filter = (a: bigint, b: bigint, op: Op) => !isExpressionHeldOut(a, op, b);
  const mix =
    preset.mix === "ninja" ? NINJA_MIX : ({ "+": 0.35, "-": 0.2, "*": 0.3, "/": 0.15 } as const);
  const dataset = new ArithmeticDataset({
    filter,
    mix,
    hardBias: preset.hardBias,
    maxDigits: preset.curriculumStart,
  });
  const pipe = new MathDataPipeline("pretrain", dataset);
  const holdExprs = buildExpressionHoldout(256, FAIR_HOLDOUT_PCT, preset.curriculumEnd);
  const heldBatch = batchFromExprs(holdExprs);
  const steps = Math.max(1, Math.round(preset.examples / preset.batch));
  const trainer = new MathTrainer();

  console.log(
    `\n=== Pretrain ${preset.label} · H=${preset.hidden} · ep=${preset.epochs} · ex=${preset.examples} · steps=${steps} · curr ${preset.curriculumStart}→${preset.curriculumEnd}${FAST ? " [FAST]" : ""} ===`,
  );
  const t0 = Date.now();
  const hist = await trainer.train(model, pipe, {
    stage: "pretrain",
    epochs: preset.epochs,
    batchSize: preset.batch,
    learningRate: preset.lr,
    stepsPerEpoch: steps,
    heldoutBatch: heldBatch,
    curriculumStartDigits: preset.curriculumStart,
    curriculumEndDigits: preset.curriculumEnd,
    onCurriculumDigits: (d) => dataset.setMaxDigits(d),
    onMetrics: (m) => {
      if (m.epoch === 1 || m.epoch === preset.epochs || m.epoch % Math.max(1, Math.floor(preset.epochs / 8)) === 0) {
        const hold = m.heldout
          ? ` holdExact=${(m.heldout.accuracy * 100).toFixed(1)}% dig=${(m.heldout.digitAccuracy * 100).toFixed(1)}%`
          : "";
        console.log(
          `  ep ${String(m.epoch).padStart(3)}  d=${m.maxDigits}  loss=${m.loss.toFixed(4)}  exact=${(m.accuracy * 100).toFixed(1)}%  digit=${(m.digitAccuracy * 100).toFixed(1)}%${hold}`,
        );
      }
    },
  });

  const finalHoldout = await evaluateModel(model, heldBatch);
  const easy = await probeAcc(model, EASY);
  const hard = await probeAcc(model, HARD);
  const curriculum = await probeAcc(model, [...CURRICULUM]);
  const byLength: Record<string, number> = {};
  for (const bucket of buildLengthProbes(preset.curriculumEnd, 24)) {
    byLength[String(bucket.digits)] = await probeAcc(model, bucket.exprs);
  }
  const sec = ((Date.now() - t0) / 1000).toFixed(1);

  console.log(
    `  done in ${sec}s · holdExact=${(finalHoldout.accuracy * 100).toFixed(1)}% dig=${(finalHoldout.digitAccuracy * 100).toFixed(1)}% · easy=${(easy * 100).toFixed(0)}% · hard=${(hard * 100).toFixed(0)}% · curr=${(curriculum * 100).toFixed(0)}%`,
  );
  console.log(
    `  byLength: ${Object.entries(byLength)
      .map(([d, a]) => `${d}=${(a * 100).toFixed(0)}%`)
      .join(" · ")}`,
  );

  const meta = buildCheckpointMeta({
    id: id as CheckpointId,
    label: preset.label,
    blurb: preset.blurb,
    hyperparams: {
      hidden: preset.hidden,
      epochs: preset.epochs,
      lr: preset.lr,
      examples: preset.examples,
      batch: preset.batch,
      mix: preset.mix,
      hardBias: preset.hardBias,
      heldout: true,
      curriculumStart: preset.curriculumStart,
      curriculumEnd: preset.curriculumEnd,
      operandDigits: OPERAND_DIGITS,
      answerDigits: ANSWER_DIGITS,
    },
    holdout: {
      pct: FAIR_HOLDOUT_PCT,
      n: holdExprs.length,
      seed: FAIR_HOLDOUT_SEED,
    },
    curves: hist,
    finalHoldout,
    probes: { easy, hard, curriculum, byLength },
    paramCount: TinyMathMlp.paramCount(preset.hidden),
  });

  const ckpt = packCheckpoint(meta, model);
  mkdirSync(OUT_DIR, { recursive: true });
  const path = join(OUT_DIR, `${id}.json`);
  writeFileSync(path, JSON.stringify(ckpt));
  const sizeKb = (Buffer.byteLength(JSON.stringify(ckpt)) / 1024).toFixed(0);
  console.log(`  wrote ${path} (${sizeKb} KB)`);

  return {
    id: id as CheckpointId,
    label: preset.label,
    blurb: preset.blurb,
    path: `checkpoints/${id}.json`,
    hidden: preset.hidden,
    epochs: preset.epochs,
    paramCount: TinyMathMlp.paramCount(preset.hidden),
    finalHoldoutAcc: finalHoldout.accuracy,
    curriculumEnd: preset.curriculumEnd,
  };
}

async function main() {
  const only = process.env.CHECKPOINT_ONLY as ModeId | undefined;
  const order: ModeId[] = only ? [only] : ["simple", "mid", "ninja"];
  mkdirSync(OUT_DIR, { recursive: true });

  const models: CheckpointManifest["models"] = [];
  if (only && existsSync(join(OUT_DIR, "manifest.json"))) {
    try {
      const prev = JSON.parse(readFileSync(join(OUT_DIR, "manifest.json"), "utf8")) as CheckpointManifest;
      for (const m of prev.models) {
        if (m.id !== only) models.push(m);
      }
    } catch {
      /* ignore */
    }
  }

  for (const id of order) {
    const entry = await trainOne(id);
    const idx = models.findIndex((m) => m.id === entry.id);
    if (idx >= 0) models[idx] = entry;
    else models.push(entry);
  }

  models.sort((a, b) => ["simple", "mid", "ninja"].indexOf(a.id) - ["simple", "mid", "ninja"].indexOf(b.id));
  const manifest: CheckpointManifest = { version: CHECKPOINT_VERSION, models };
  writeFileSync(join(OUT_DIR, "manifest.json"), JSON.stringify(manifest, null, 2));
  console.log(`\nManifest → ${join(OUT_DIR, "manifest.json")}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
