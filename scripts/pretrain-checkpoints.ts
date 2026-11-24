/**
 * Offline CPU pretrain → public/checkpoints/{simple,mid,ninja}.json
 *
 * Run: npx tsx scripts/pretrain-checkpoints.ts
 * Optional: CHECKPOINT_ONLY=simple npx tsx scripts/pretrain-checkpoints.ts
 */
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { CpuBackend } from "../src/webgpu";
import { TinyMathMlp } from "../src/model";
import {
  ArithmeticDataset,
  CURRICULUM,
  MathDataPipeline,
  NINJA_MIX,
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
  [47, "*", 19],
];

async function probeAcc(
  model: TinyMathMlp,
  probes: Array<readonly [number, Op, number]>,
): Promise<number> {
  let ok = 0;
  let n = 0;
  for (const [a, op, b] of probes) {
    const truth = computeAnswer(a, op, b);
    if (truth >= 1000) continue;
    n++;
    const { answer } = await model.predict(encodeFeatures(a, op, b));
    if (answer === truth) ok++;
  }
  return n === 0 ? 0 : ok / n;
}

async function trainOne(id: ModeId): Promise<CheckpointManifest["models"][number]> {
  const preset = MODE_PRESETS[id];
  const gpu = new CpuBackend();
  const model = new TinyMathMlp(gpu, preset.hidden);
  const filter = (a: number, b: number, op: Op) => !isExpressionHeldOut(a, op, b);
  const mix =
    preset.mix === "ninja" ? NINJA_MIX : ({ "+": 0.35, "-": 0.2, "*": 0.3, "/": 0.15 } as const);
  const pipe = new MathDataPipeline(
    "pretrain",
    new ArithmeticDataset({ filter, mix, hardBias: preset.hardBias }),
  );
  const holdExprs = buildExpressionHoldout(200, FAIR_HOLDOUT_PCT);
  const heldBatch = batchFromExprs(holdExprs);
  const steps = Math.max(1, Math.round(preset.examples / preset.batch));
  const trainer = new MathTrainer();

  console.log(
    `\n=== Pretrain ${preset.label} · H=${preset.hidden} · ep=${preset.epochs} · ex=${preset.examples} · steps=${steps} ===`,
  );
  const t0 = Date.now();
  const hist = await trainer.train(model, pipe, {
    stage: "pretrain",
    epochs: preset.epochs,
    batchSize: preset.batch,
    learningRate: preset.lr,
    stepsPerEpoch: steps,
    heldoutBatch: heldBatch,
    onMetrics: (m) => {
      if (m.epoch === 1 || m.epoch === preset.epochs || m.epoch % 10 === 0) {
        const hold = m.heldout
          ? ` holdAcc=${(m.heldout.accuracy * 100).toFixed(1)}%`
          : "";
        console.log(
          `  ep ${String(m.epoch).padStart(3)}  loss=${m.loss.toFixed(4)}  acc=${(m.accuracy * 100).toFixed(1)}%${hold}`,
        );
      }
    },
  });

  const finalHoldout = await evaluateModel(model, heldBatch);
  const easy = await probeAcc(model, EASY);
  const hard = await probeAcc(model, HARD);
  const curriculum = await probeAcc(model, [...CURRICULUM]);
  const sec = ((Date.now() - t0) / 1000).toFixed(1);

  console.log(
    `  done in ${sec}s · holdAcc=${(finalHoldout.accuracy * 100).toFixed(1)}% · easy=${(easy * 100).toFixed(0)}% · hard=${(hard * 100).toFixed(0)}% · curr=${(curriculum * 100).toFixed(0)}%`,
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
    },
    holdout: {
      pct: FAIR_HOLDOUT_PCT,
      n: holdExprs.length,
      seed: FAIR_HOLDOUT_SEED,
    },
    curves: hist,
    finalHoldout,
    probes: { easy, hard, curriculum },
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
  };
}

async function main() {
  const only = process.env.CHECKPOINT_ONLY as ModeId | undefined;
  const order: ModeId[] = only ? [only] : ["simple", "mid", "ninja"];
  mkdirSync(OUT_DIR, { recursive: true });

  const models: CheckpointManifest["models"] = [];
  // Keep prior manifest entries when training a single id.
  if (only && existsSync(join(OUT_DIR, "manifest.json"))) {
    try {
      const prev = JSON.parse(
        await import("node:fs").then((fs) => fs.readFileSync(join(OUT_DIR, "manifest.json"), "utf8")),
      ) as CheckpointManifest;
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
