/**
 * Offline digit RL on a saved checkpoint → print hold-out before/after.
 *
 *   npx tsx scripts/run-digit-rl.ts
 *   CHECKPOINT=simple RL_STEPS=80 npx tsx scripts/run-digit-rl.ts
 *
 * Optionally writes public/checkpoints/{id}-digit-rl.json when WRITE=1.
 */
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { CpuBackend } from "../src/webgpu";
import { TinyMathMlp } from "../src/model";
import {
  FAIR_HOLDOUT_PCT,
  FAIR_HOLDOUT_SEED,
  buildExpressionHoldout,
} from "../src/evals";
import {
  buildCheckpointMeta,
  loadWeights,
  packCheckpoint,
  type CheckpointFile,
  type CheckpointId,
} from "../src/checkpoint";
import { runDigitRl } from "../src/posttrain";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const CKPT_DIR = join(ROOT, "public", "checkpoints");
const id = (process.env.CHECKPOINT ?? "ninja") as CheckpointId;
const steps = Math.max(5, Number(process.env.RL_STEPS ?? 60));
const lr = Number(process.env.RL_LR ?? 0.0015);
const write = process.env.WRITE === "1";

async function main() {
  const path = join(CKPT_DIR, `${id}.json`);
  if (!existsSync(path)) {
    throw new Error(`Missing ${path} — run npm run pretrain:checkpoints first.`);
  }
  const ckpt = JSON.parse(readFileSync(path, "utf8")) as CheckpointFile;
  const gpu = new CpuBackend();
  const model = new TinyMathMlp(gpu, ckpt.hyperparams.hidden);
  loadWeights(model, ckpt.weights.params);

  const holdExprs = buildExpressionHoldout(200, FAIR_HOLDOUT_PCT);
  console.log(
    `Digit RL on ${ckpt.label} (H=${ckpt.hyperparams.hidden}) · steps=${steps} · lr=${lr} · hold n=${holdExprs.length}`,
  );

  const result = await runDigitRl(model, {
    steps,
    learningRate: lr,
    holdoutExprs: holdExprs,
    onStep: (m) => {
      if (m.step === m.steps || m.step % Math.max(1, Math.floor(m.steps / 5)) === 0) {
        console.log(
          `  step ${m.step}/${m.steps}  reward=${m.meanReward.toFixed(4)}  digit=${(m.meanDigitAcc * 100).toFixed(1)}%  exact=${(m.exactRate * 100).toFixed(1)}%  adv=${m.meanAdvantage.toFixed(4)}`,
        );
      }
    },
  });

  const before = (result.beforeHoldout.accuracy * 100).toFixed(1);
  const after = (result.afterHoldout.accuracy * 100).toFixed(1);
  const deltaPp = (result.holdoutDelta * 100).toFixed(1);
  console.log(`HOLD-OUT before ${before}% → after ${after}% (Δ ${Number(deltaPp) >= 0 ? "+" : ""}${deltaPp} pp)`);
  console.log(`mean reward ${result.meanReward.toFixed(4)}`);

  if (write) {
    mkdirSync(CKPT_DIR, { recursive: true });
    const meta = buildCheckpointMeta({
      id: ckpt.id,
      label: `${ckpt.label} + digit RL`,
      blurb: `Digit RL (${steps} steps) on ${ckpt.label}. Hold-out ${before}% → ${after}%.`,
      hyperparams: ckpt.hyperparams,
      holdout: { pct: FAIR_HOLDOUT_PCT, n: result.afterHoldout.n, seed: FAIR_HOLDOUT_SEED },
      curves: ckpt.curves,
      finalHoldout: result.afterHoldout,
      probes: ckpt.probes,
      paramCount: TinyMathMlp.paramCount(model.hidden),
    });
    const out = packCheckpoint(meta, model);
    out.postTrain = {
      kind: "digit-rl",
      steps: result.steps,
      meanReward: result.meanReward,
      beforeHoldout: result.beforeHoldout,
      afterHoldout: result.afterHoldout,
      holdoutDelta: result.holdoutDelta,
      baseId: ckpt.id,
    };
    const outPath = join(CKPT_DIR, `${id}-digit-rl.json`);
    writeFileSync(outPath, JSON.stringify(out));
    console.log(`Wrote ${outPath}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
