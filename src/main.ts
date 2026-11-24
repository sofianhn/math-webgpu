import { WebGpuBackend } from "./webgpu";
import { DEFAULT_HIDDEN, TinyMathMlp } from "./model";
import {
  ArithmeticDataset,
  CURRICULUM,
  MAX_OPERAND,
  MathDataPipeline,
  NINJA_MIX,
  OPS,
  TOKENS_PER_EXAMPLE,
  computeAnswer,
  encodeFeatures,
  formatExpr,
  inDomain,
  isThreeDigitOrBeyond,
  type Op,
} from "./data";
import { MathTrainer } from "./trainer";
import type { TrainMetrics } from "./types";
import { renderCurves, renderDataFeed, renderNormBars, renderTopK } from "./viz";
import {
  FAIR_HOLDOUT_PCT,
  batchFromExprs,
  buildExpressionHoldout,
  evaluateModel,
  isExpressionHeldOut,
} from "./evals";
import { DEFAULT_MODE, MODE_PRESETS, type ModeId, type ModePreset } from "./presets";
import {
  fetchCheckpoint,
  fetchManifest,
  loadWeights,
  type CheckpointFile,
  type CheckpointId,
} from "./checkpoint";
import { describePostTrain, runPostTrain } from "./posttrain";

const statusEl = document.querySelector("#status")!;
const controlsEl = document.querySelector<HTMLElement>("#controls")!;
const dataEl = document.querySelector<HTMLElement>("#data")!;
const dataFeedEl = document.querySelector<HTMLElement>("#data-feed")!;
const holdFeedEl = document.querySelector<HTMLElement>("#hold-feed")!;
const internalsEl = document.querySelector<HTMLElement>("#internals")!;
const logEl = document.querySelector("#log")!;
const probeEl = document.querySelector<HTMLElement>("#probe")!;
const posttrainEl = document.querySelector<HTMLElement>("#posttrain-note")!;
const trainBtn = document.querySelector<HTMLButtonElement>("#train")!;
const evalBtn = document.querySelector<HTMLButtonElement>("#eval")!;
const loadCkptBtn = document.querySelector<HTMLButtonElement>("#load-ckpt")!;
const predictBtn = document.querySelector<HTMLButtonElement>("#predict")!;
const posttrainBtn = document.querySelector<HTMLButtonElement>("#posttrain-stub")!;
const predictionEl = document.querySelector("#prediction")!;
const oodEl = document.querySelector<HTMLElement>("#ood")!;
const curvesEl = document.querySelector<SVGSVGElement>("#curves")!;
const normsEl = document.querySelector<HTMLElement>("#norms")!;
const topkEl = document.querySelector<HTMLElement>("#topk")!;
const smartnessEl = document.querySelector("#smartness")!;
const evalLineEl = document.querySelector("#eval-line")!;
const progressEl = document.querySelector("#progress")!;
const progressFill = document.querySelector<HTMLElement>("#progress-fill")!;
const paramCountEl = document.querySelector("#param-count")!;
const tokenCountEl = document.querySelector("#token-count")!;
const modeBlurbEl = document.querySelector("#mode-blurb")!;
const ckptMetaEl = document.querySelector("#ckpt-meta")!;
const epochsInput = document.querySelector<HTMLInputElement>("#epochs")!;
const batchInput = document.querySelector<HTMLInputElement>("#batch")!;
const examplesInput = document.querySelector<HTMLInputElement>("#examples")!;
const lrInput = document.querySelector<HTMLInputElement>("#lr")!;
const hiddenInput = document.querySelector<HTMLInputElement>("#hidden")!;
const heldoutInput = document.querySelector<HTMLInputElement>("#heldout")!;
const aInput = document.querySelector<HTMLInputElement>("#a")!;
const bInput = document.querySelector<HTMLInputElement>("#b")!;

let activeMode: ModeId | "custom" = DEFAULT_MODE;
let loadedCkpt: CheckpointFile | null = null;
let modelReady = false;

function paint(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
}

function log(line: string) {
  logEl.textContent += line + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

function formatPpl(p: number): string {
  if (!Number.isFinite(p)) return "∞";
  if (p >= 1e6) return p.toExponential(2);
  if (p >= 1000) return p.toFixed(0);
  return p.toFixed(2);
}

function applyMode(mode: ModePreset) {
  activeMode = mode.id;
  hiddenInput.value = String(mode.hidden);
  epochsInput.value = String(mode.epochs);
  lrInput.value = String(mode.lr);
  examplesInput.value = String(mode.examples);
  batchInput.value = String(mode.batch);
  modeBlurbEl.textContent = mode.blurb;
  document.querySelectorAll<HTMLButtonElement>("[data-ckpt]").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.ckpt === mode.id);
  });
  refreshCounts();
}

function markCustom() {
  activeMode = "custom";
  document.querySelectorAll<HTMLButtonElement>("[data-ckpt]").forEach((btn) => {
    btn.classList.remove("is-active");
  });
  modeBlurbEl.textContent = "Custom knobs — not a saved Simple/Mid/Ninja model.";
  ckptMetaEl.textContent = "";
}

function refreshCounts() {
  const hidden = Math.max(16, Math.min(512, Number(hiddenInput.value) || DEFAULT_HIDDEN));
  const batch = Math.max(4, Number(batchInput.value) || 64);
  let examples = Math.max(batch, Number(examplesInput.value) || 2560);
  examples = Math.max(batch, Math.round(examples / batch) * batch);
  examplesInput.value = String(examples);
  const steps = examples / batch;
  const tokens = examples * TOKENS_PER_EXAMPLE;
  const params = TinyMathMlp.paramCount(hidden);
  const modeTag = activeMode === "custom" ? "custom" : activeMode;
  paramCountEl.textContent = `Parameters ≈ ${params.toLocaleString()}  · mode=${modeTag}  (digit heads, no flat W2)`;
  tokenCountEl.textContent = `Tokens / epoch ≈ ${tokens.toLocaleString()}  (${examples.toLocaleString()} examples × ${TOKENS_PER_EXAMPLE} · ${steps} steps × batch ${batch})`;
}

function paintCkptMeta(ckpt: CheckpointFile) {
  const h = ckpt.hyperparams;
  const hold = ckpt.finalHoldout;
  const probes = ckpt.probes
    ? ` · probes easy ${(ckpt.probes.easy * 100).toFixed(0)}% / hard ${(ckpt.probes.hard * 100).toFixed(0)}% / curr ${(ckpt.probes.curriculum * 100).toFixed(0)}%`
    : "";
  ckptMetaEl.textContent = `Loaded ${ckpt.label}: H=${h.hidden} · ${h.epochs} ep · lr=${h.lr} · ${h.examples.toLocaleString()} ex/ep · batch ${h.batch} · final hold-out ${(hold.accuracy * 100).toFixed(1)}% (n=${hold.n})${probes}`;
}

async function curriculumScore(model: TinyMathMlp): Promise<{ ok: number; total: number }> {
  let ok = 0;
  for (const [a, op, b] of CURRICULUM) {
    const truth = computeAnswer(a, op, b);
    if (truth >= 1000) continue;
    const { answer } = await model.predict(encodeFeatures(a, op, b));
    if (answer === truth) ok++;
  }
  return { ok, total: CURRICULUM.length };
}

async function boot() {
  try {
    const gpu = await WebGpuBackend.create();
    statusEl.innerHTML = `<p><strong>WebGPU ready.</strong> Load a <strong>saved</strong> Simple / Mid / Ninja model for instant curves, or Train New. Domain <strong>0–${MAX_OPERAND}</strong>.</p>`;
    controlsEl.hidden = false;
    dataEl.hidden = false;
    internalsEl.hidden = false;
    probeEl.hidden = false;
    posttrainEl.hidden = false;
    renderCurves(curvesEl, []);
    applyMode(MODE_PRESETS[DEFAULT_MODE]);

    const holdExprs = buildExpressionHoldout(200, FAIR_HOLDOUT_PCT);
    renderDataFeed(
      holdFeedEl,
      holdExprs.slice(0, 8).map(([a, op, b]) => {
        const ans = computeAnswer(a, op, b);
        return `${formatExpr(a, op, b, ans)}  ·  fair held-out (frozen expression)`;
      }),
      `Frozen fair held-out bank: ${holdExprs.length} exact expressions (~${FAIR_HOLDOUT_PCT}% slice).`,
    );

    let model = new TinyMathMlp(gpu, MODE_PRESETS.ninja.hidden);
    const trainer = new MathTrainer();
    let history: TrainMetrics[] = [];

    const smoke = await gpu.matmul(
      new Float32Array([1, 2, 3, 4]),
      new Float32Array([5, 6, 7, 8]),
      2,
      2,
      2,
    );
    const expected = [19, 22, 43, 50];
    log(
      expected.every((v, i) => Math.abs(smoke[i]! - v) < 1e-3)
        ? "GPU matmul smoke-test passed (2×2)."
        : `GPU matmul mismatch: ${Array.from(smoke)}`,
    );

    const manifest = await fetchManifest();
    if (manifest?.models.length) {
      const bits = manifest.models
        .map((m) => `${m.label} hold=${(m.finalHoldoutAcc * 100).toFixed(0)}%`)
        .join(" · ");
      log(`Checkpoints available: ${bits}`);
    } else {
      log("No checkpoint manifest found — Train New still works; Load saved needs public/checkpoints/.");
    }

    const setBusy = (busy: boolean) => {
      trainBtn.disabled = busy;
      evalBtn.disabled = busy || !modelReady;
      loadCkptBtn.disabled = busy;
      predictBtn.disabled = busy || !modelReady;
    };

    const applyCheckpoint = async (ckpt: CheckpointFile, opts?: { silent?: boolean }) => {
      const hp = ckpt.hyperparams;
      applyMode({
        id: ckpt.id,
        label: ckpt.label,
        blurb: ckpt.blurb,
        hidden: hp.hidden,
        epochs: hp.epochs,
        lr: hp.lr,
        examples: hp.examples,
        batch: hp.batch,
        mix: hp.mix,
        hardBias: hp.hardBias,
      });
      model = new TinyMathMlp(gpu, hp.hidden);
      loadWeights(model, ckpt.weights.params);
      loadedCkpt = ckpt;
      history = ckpt.curves.slice();
      modelReady = true;
      paintCkptMeta(ckpt);
      renderCurves(curvesEl, history, {
        title: `${ckpt.label} (saved) · H=${hp.hidden} · hold ${(ckpt.finalHoldout.accuracy * 100).toFixed(0)}%`,
      });
      progressFill.style.width = "100%";
      progressEl.textContent = `Loaded ${ckpt.label} · ${ckpt.curves.length} epochs of curves · hold-out ${(ckpt.finalHoldout.accuracy * 100).toFixed(1)}%`;
      evalLineEl.textContent = `Saved held-out: perplexity ${formatPpl(ckpt.finalHoldout.perplexity)} · acc ${(ckpt.finalHoldout.accuracy * 100).toFixed(1)}% on ${ckpt.finalHoldout.n} exprs`;
      const score = await curriculumScore(model);
      smartnessEl.textContent = `Curriculum ${score.ok}/${score.total} (${((score.ok / score.total) * 100).toFixed(0)}%) — from saved ${ckpt.label}`;
      normsEl.innerHTML = `<p class="norm-intro">Loaded ${ckpt.arch.paramCount.toLocaleString()} parameters from checkpoint (train from scratch to see before→after norm bars).</p>`;
      if (!opts?.silent) {
        log(
          `Loaded checkpoint ${ckpt.id} (${ckpt.arch.paramCount.toLocaleString()} params, ${ckpt.curves.length} epoch curves).`,
        );
      }
      evalBtn.disabled = false;
      predictBtn.disabled = false;
    };

    const loadSelected = async () => {
      const id = (activeMode === "custom" ? DEFAULT_MODE : activeMode) as CheckpointId;
      setBusy(true);
      try {
        const ckpt = await fetchCheckpoint(id);
        await applyCheckpoint(ckpt);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        log(`Load failed: ${message}`);
        ckptMetaEl.textContent = `Could not load ${id}: ${message}`;
      } finally {
        setBusy(false);
      }
    };

    document.querySelectorAll<HTMLButtonElement>("[data-ckpt]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const key = btn.dataset.ckpt as ModeId;
        applyMode(MODE_PRESETS[key]);
        void loadSelected();
      });
    });
    loadCkptBtn.addEventListener("click", () => void loadSelected());

    for (const el of [hiddenInput, batchInput, examplesInput, epochsInput, lrInput]) {
      el.addEventListener("change", () => {
        markCustom();
        refreshCounts();
      });
    }

    const previewFilter = (a: number, b: number, op: Op) => !isExpressionHeldOut(a, op, b);
    const previewPipe = new MathDataPipeline(
      "pretrain",
      new ArithmeticDataset({ filter: previewFilter, mix: NINJA_MIX, hardBias: 0.45 }),
    );
    renderDataFeed(
      dataFeedEl,
      previewPipe.nextBatch(6).texts ?? [],
      "Preview of train-distribution rows (Ninja mix favors × / carries; fair held-out excluded):",
    );

    // Auto-load default Ninja checkpoint when present.
    await loadSelected();

    trainBtn.addEventListener("click", async () => {
      setBusy(true);
      logEl.textContent = "";
      refreshCounts();

      const epochs = Number(epochsInput.value);
      const batchSize = Math.max(4, Number(batchInput.value) || 64);
      const examples = Math.max(batchSize, Number(examplesInput.value) || 2560);
      const stepsPerEpoch = Math.max(1, Math.round(examples / batchSize));
      const learningRate = Number(lrInput.value);
      const hidden = Math.max(16, Math.min(512, Number(hiddenInput.value) || DEFAULT_HIDDEN));
      hiddenInput.value = String(hidden);
      const useHeldout = heldoutInput.checked;
      const modeLabel = activeMode;

      const trainFilter = useHeldout
        ? (a: number, b: number, op: Op) => !isExpressionHeldOut(a, op, b)
        : undefined;
      const useNinjaMix = modeLabel === "ninja" || modeLabel === "mid" || modeLabel === "custom";
      const hardBias =
        modeLabel === "ninja"
          ? MODE_PRESETS.ninja.hardBias
          : modeLabel === "mid"
            ? MODE_PRESETS.mid.hardBias
            : modeLabel === "simple"
              ? MODE_PRESETS.simple.hardBias
              : 0.45;
      const pipeline = new MathDataPipeline(
        "pretrain",
        new ArithmeticDataset({
          filter: trainFilter,
          mix: useNinjaMix
            ? NINJA_MIX
            : { "+": 0.35, "-": 0.2, "*": 0.3, "/": 0.15 },
          hardBias,
        }),
      );
      const heldoutBatch = useHeldout ? batchFromExprs(holdExprs) : undefined;

      model = new TinyMathMlp(gpu, hidden);
      const beforeNorms = model.weightNorms();
      history = [];
      loadedCkpt = null;
      modelReady = false;
      renderCurves(curvesEl, history);
      progressFill.style.width = "0%";
      progressEl.textContent = `Starting ${modeLabel}… 0 / ${epochs}`;
      evalLineEl.textContent = "";
      ckptMetaEl.textContent = `Training new ${modeLabel} from scratch (not using saved weights).`;
      const fb0 = gpu.cpuFallbackCount;

      log(
        `Train New [${modeLabel}] — digit heads · H=${hidden} (~${TinyMathMlp.paramCount(hidden).toLocaleString()} params) · ${examples}/epoch · ${stepsPerEpoch} steps × batch ${batchSize} · ~${(examples * TOKENS_PER_EXAMPLE).toLocaleString()} tokens/epoch · lr=${learningRate} · held-out=${useHeldout ? `fair ~${FAIR_HOLDOUT_PCT}%` : "off"}`,
      );
      if (modeLabel === "ninja") {
        log("Ninja can take a few minutes on phones — that’s the strong-math setting.");
      }
      await paint();

      await trainer.train(model, pipeline, {
        stage: "pretrain",
        epochs,
        batchSize,
        learningRate,
        stepsPerEpoch,
        tokensPerExample: TOKENS_PER_EXAMPLE,
        heldoutBatch,
        onMetrics: async (m) => {
          history.push(m);
          progressFill.style.width = `${((m.epoch / epochs) * 100).toFixed(1)}%`;
          progressEl.textContent = `Epoch ${m.epoch}/${epochs} · train loss ${m.loss.toFixed(3)} · ppl ${formatPpl(m.perplexity)} · acc ${(m.accuracy * 100).toFixed(1)}% · ${m.tokens.toLocaleString()} tokens`;
          let line = `epoch ${String(m.epoch).padStart(3)}  loss=${m.loss.toFixed(4)}  ppl=${formatPpl(m.perplexity)}  acc=${(m.accuracy * 100).toFixed(1)}%`;
          if (m.heldout) {
            line += `  |  HOLD loss=${m.heldout.loss.toFixed(4)}  ppl=${formatPpl(m.heldout.perplexity)}  acc=${(m.heldout.accuracy * 100).toFixed(1)}%`;
            evalLineEl.textContent = `Held-out (unseen exprs): perplexity ${formatPpl(m.heldout.perplexity)} · acc ${(m.heldout.accuracy * 100).toFixed(1)}% on ${m.heldout.n} exprs`;
          }
          log(line);
          renderCurves(curvesEl, history, {
            title: `${modeLabel} · H=${hidden} · ${examples} ex/ep · lr=${learningRate}`,
          });
          const live = pipeline.nextBatch(6);
          renderDataFeed(dataFeedEl, live.texts ?? [], `Live train sample @ epoch ${m.epoch}:`);
          if (m.epoch === epochs || m.epoch % Math.max(1, Math.floor(epochs / 5)) === 0) {
            const score = await curriculumScore(model);
            smartnessEl.textContent = `Curriculum @ epoch ${m.epoch}: ${score.ok}/${score.total} (${(
              (score.ok / score.total) *
              100
            ).toFixed(0)}%)`;
          }
          await paint();
        },
      });

      renderNormBars(normsEl, beforeNorms, model.weightNorms());
      const score = await curriculumScore(model);
      smartnessEl.textContent = `Curriculum ${score.ok}/${score.total} (${((score.ok / score.total) * 100).toFixed(0)}%) — compare held-out acc.`;
      progressEl.textContent = `Done [${modeLabel}] · ${epochs} epochs · final train ppl ${formatPpl(history.at(-1)?.perplexity ?? NaN)}`;
      progressFill.style.width = "100%";
      modelReady = true;
      const fb = gpu.cpuFallbackCount - fb0;
      log(`Train New finished.${fb ? ` (CPU matmul fallback ${fb}×)` : ""}`);

      const holdProbe =
        holdExprs.find(([a, op, b]) => op === "+" && computeAnswer(a, op, b) < 1000) ?? holdExprs[0];
      const probes: Array<readonly [number, Op, number]> = [
        [12, "+", 34],
        [47, "+", 38],
        [17, "*", 19],
        [47, "*", 19],
        [25, "*", 25],
        holdProbe ?? [58, "+", 67],
      ];
      for (const [a, op, b] of probes) {
        const truth = computeAnswer(a, op, b);
        const { answer } = await model.predict(encodeFeatures(a, op, b));
        const tag = isExpressionHeldOut(a, op, b) ? "HOLD" : "probe";
        log(`probe [${tag}] ${formatExpr(a, op, b)} → ${answer} (want ${truth})${answer === truth ? " ✓" : " ✗"}`);
      }

      setBusy(false);
    });

    evalBtn.addEventListener("click", async () => {
      if (!modelReady) return;
      setBusy(true);
      const pipeline = new MathDataPipeline(
        "pretrain",
        new ArithmeticDataset({
          filter: heldoutInput.checked ? (a, b, op) => !isExpressionHeldOut(a, op, b) : undefined,
          mix: NINJA_MIX,
          hardBias: 0.45,
        }),
      );
      const { loss, accuracy, perplexity } = await trainer.evaluate(model, pipeline, 15);
      log(`eval train-dist  loss=${loss.toFixed(4)}  ppl=${formatPpl(perplexity)}  acc=${(accuracy * 100).toFixed(1)}%`);
      if (heldoutInput.checked) {
        const held = await evaluateModel(model, batchFromExprs(holdExprs));
        log(
          `eval HOLD       loss=${held.loss.toFixed(4)}  ppl=${formatPpl(held.perplexity)}  acc=${(held.accuracy * 100).toFixed(1)}%  n=${held.n}`,
        );
        evalLineEl.textContent = `Live held-out: perplexity ${formatPpl(held.perplexity)} · acc ${(held.accuracy * 100).toFixed(1)}% on ${held.n} exprs`;
      }
      setBusy(false);
    });

    const runPredict = async () => {
      if (!modelReady) {
        predictionEl.textContent = "Load a saved model or Train New first.";
        return;
      }
      const a = Number(aInput.value);
      const b = Number(bInput.value);
      const op = (document.querySelector("#op") as HTMLSelectElement).value as Op;
      if (!OPS.includes(op)) return;

      if (!inDomain(a, b)) {
        oodEl.hidden = false;
        const offender = isThreeDigitOrBeyond(a) ? a : b;
        oodEl.textContent = `${formatExpr(a, op, b)} is OOD (need 0–${MAX_OPERAND}). ${offender} is outside the two-digit encoder.`;
        predictionEl.textContent = "";
        topkEl.innerHTML = "";
        return;
      }
      oodEl.hidden = true;
      const truth = computeAnswer(a, op, b);
      if (truth >= 1000) {
        oodEl.hidden = false;
        oodEl.textContent = `Truth ${truth} exceeds 1000 answer classes.`;
        return;
      }
      const { answer, probs } = await model.predict(encodeFeatures(a, op, b));
      const tag = isExpressionHeldOut(a, op, b) ? " [fair held-out]" : "";
      predictionEl.textContent = `${formatExpr(a, op, b)} → model says ${answer} (truth ${truth})${
        answer === truth ? " ✓" : " ✗"
      }${tag}`;
      renderTopK(topkEl, probs, 5);
    };

    predictBtn.addEventListener("click", () => void runPredict());
    aInput.addEventListener("change", () => void runPredict());
    bInput.addEventListener("change", () => void runPredict());

    posttrainBtn.addEventListener("click", () => {
      log(`Post-train stub: ${describePostTrain("digit-rl")}`);
      void runPostTrain(model, { kind: "digit-rl" }).catch((err) => {
        log(err instanceof Error ? err.message : String(err));
      });
    });
    // Keep disabled — stub only — but allow a click via title; enable for discoverability:
    posttrainBtn.disabled = false;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    statusEl.innerHTML = `<p><strong>WebGPU failed.</strong> ${message}</p>`;
    log(message);
  }
}

boot();
