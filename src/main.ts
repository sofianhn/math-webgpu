import { WebGpuBackend } from "./webgpu";
import { DEFAULT_HIDDEN, TinyMathMlp } from "./model";
import {
  ANSWER_DIGITS,
  ArithmeticDataset,
  CURRICULUM,
  MAX_OPERAND,
  MathDataPipeline,
  NINJA_MIX,
  OPERAND_DIGITS,
  OPS,
  TOKENS_PER_EXAMPLE,
  computeAnswer,
  encodeFeatures,
  formatExpr,
  inDomain,
  isBeyondOperandDigits,
  type Op,
} from "./data";
import { MathTrainer } from "./trainer";
import type { TrainMetrics } from "./types";
import { renderCurves, renderDataFeed, renderDigitPlaces, renderNormBars } from "./viz";
import {
  FAIR_HOLDOUT_PCT,
  FAIR_HOLDOUT_SEED,
  batchFromExprs,
  buildExpressionHoldout,
  evaluateModel,
  isExpressionHeldOut,
} from "./evals";
import { DEFAULT_MODE, MODE_PRESETS, type ModeId, type ModePreset } from "./presets";
import {
  buildCheckpointMeta,
  downloadCheckpoint,
  fetchCheckpoint,
  fetchManifest,
  loadWeights,
  packCheckpoint,
  type CheckpointFile,
  type CheckpointId,
} from "./checkpoint";
import {
  describePostTrain,
  runPostTrain,
  type DigitRlResult,
  type PostTrainKind,
} from "./posttrain";

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
const digitRlBtn = document.querySelector<HTMLButtonElement>("#digit-rl")!;
const saveCkptBtn = document.querySelector<HTMLButtonElement>("#save-ckpt")!;
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
const evalStatusEl = document.querySelector<HTMLElement>("#eval-status")!;
const rlStatusEl = document.querySelector<HTMLElement>("#rl-status")!;
const rlDeltaEl = document.querySelector<HTMLElement>("#rl-delta")!;
const epochsInput = document.querySelector<HTMLInputElement>("#epochs")!;
const batchInput = document.querySelector<HTMLInputElement>("#batch")!;
const examplesInput = document.querySelector<HTMLInputElement>("#examples")!;
const lrInput = document.querySelector<HTMLInputElement>("#lr")!;
const hiddenInput = document.querySelector<HTMLInputElement>("#hidden")!;
const heldoutInput = document.querySelector<HTMLInputElement>("#heldout")!;
const rlStepsInput = document.querySelector<HTMLInputElement>("#rl-steps")!;
const rlLrInput = document.querySelector<HTMLInputElement>("#rl-lr")!;
const postKindSel = document.querySelector<HTMLSelectElement>("#post-kind")!;
const aInput = document.querySelector<HTMLInputElement>("#a")!;
const bInput = document.querySelector<HTMLInputElement>("#b")!;

let activeMode: ModeId | "custom" = DEFAULT_MODE;
let loadedCkpt: CheckpointFile | null = null;
let modelReady = false;
let lastRlResult: DigitRlResult | null = null;
let postRlCkpt: CheckpointFile | null = null;

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

function parseOperand(raw: string): bigint | null {
  const s = raw.trim().replace(/[,\s_]/g, "");
  if (!/^\d+$/.test(s)) return null;
  try {
    return BigInt(s);
  } catch {
    return null;
  }
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
  const hidden = Math.max(32, Math.min(768, Number(hiddenInput.value) || DEFAULT_HIDDEN));
  const batch = Math.max(4, Number(batchInput.value) || 64);
  let examples = Math.max(batch, Number(examplesInput.value) || 8192);
  examples = Math.max(batch, Math.round(examples / batch) * batch);
  examplesInput.value = String(examples);
  const steps = examples / batch;
  const tokens = examples * TOKENS_PER_EXAMPLE;
  const params = TinyMathMlp.paramCount(hidden);
  const modeTag = activeMode === "custom" ? "custom" : activeMode;
  paramCountEl.textContent = `Parameters ≈ ${params.toLocaleString()}  · mode=${modeTag}  (${OPERAND_DIGITS}-digit in · ${ANSWER_DIGITS} heads)`;
  tokenCountEl.textContent = `Tokens / epoch ≈ ${tokens.toLocaleString()}  (${examples.toLocaleString()} examples × ${TOKENS_PER_EXAMPLE} · ${steps} steps × batch ${batch})`;
}

function paintCkptMeta(ckpt: CheckpointFile) {
  const h = ckpt.hyperparams;
  const hold = ckpt.finalHoldout;
  const dig = hold.digitAccuracy != null ? ` · digit ${(hold.digitAccuracy * 100).toFixed(1)}%` : "";
  const probes = ckpt.probes
    ? ` · probes easy ${(ckpt.probes.easy * 100).toFixed(0)}% / hard ${(ckpt.probes.hard * 100).toFixed(0)}% / curr ${(ckpt.probes.curriculum * 100).toFixed(0)}%`
    : "";
  ckptMetaEl.textContent = `Loaded ${ckpt.label}: H=${h.hidden} · ${h.epochs} ep · curriculum ${h.curriculumStart}→${h.curriculumEnd} digits · final hold-out ${(hold.accuracy * 100).toFixed(1)}%${dig} (n=${hold.n})${probes}`;
}

async function curriculumScore(model: TinyMathMlp): Promise<{ ok: number; total: number }> {
  let ok = 0;
  for (const [a, op, b] of CURRICULUM) {
    const truth = computeAnswer(a, op, b);
    const { answer } = await model.predict(encodeFeatures(a, op, b));
    if (answer === truth) ok++;
  }
  return { ok, total: CURRICULUM.length };
}

async function boot() {
  try {
    const gpu = await WebGpuBackend.create();
    statusEl.innerHTML = `<p><strong>WebGPU ready.</strong> Nine-digit place-value MLP (operands 0–${MAX_OPERAND.toLocaleString()}). Load a saved model or Train New.</p>`;
    controlsEl.hidden = false;
    dataEl.hidden = false;
    internalsEl.hidden = false;
    probeEl.hidden = false;
    posttrainEl.hidden = false;
    renderCurves(curvesEl, []);
    applyMode(MODE_PRESETS[DEFAULT_MODE]);

    const holdExprs = buildExpressionHoldout(256, FAIR_HOLDOUT_PCT, OPERAND_DIGITS);
    renderDataFeed(
      holdFeedEl,
      holdExprs.slice(0, 8).map(([a, op, b]) => {
        const ans = computeAnswer(a, op, b);
        return `${formatExpr(a, op, b, ans)}  ·  fair held-out (frozen expression)`;
      }),
      `Frozen fair held-out bank: ${holdExprs.length} exact expressions (~${FAIR_HOLDOUT_PCT}% slice, lengths 1–${OPERAND_DIGITS}).`,
    );

    let model = new TinyMathMlp(gpu, MODE_PRESETS.ninja.hidden);
    const trainer = new MathTrainer();
    let history: TrainMetrics[] = [];
    let trainDataset: ArithmeticDataset | null = null;

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

    const setRlStatus = (text: string, kind: "" | "ok" | "error" = "") => {
      rlStatusEl.textContent = text;
      rlStatusEl.classList.toggle("is-ok", kind === "ok");
      rlStatusEl.classList.toggle("is-error", kind === "error");
    };

    const setBusy = (busy: boolean) => {
      trainBtn.disabled = busy;
      evalBtn.disabled = busy || !modelReady;
      loadCkptBtn.disabled = busy;
      predictBtn.disabled = busy || !modelReady;
      digitRlBtn.disabled = busy || !modelReady;
      saveCkptBtn.disabled = busy || !postRlCkpt;
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
        curriculumStart: hp.curriculumStart ?? 1,
        curriculumEnd: hp.curriculumEnd ?? OPERAND_DIGITS,
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
      evalLineEl.textContent = `Saved held-out: perplexity ${formatPpl(ckpt.finalHoldout.perplexity)} · exact ${(ckpt.finalHoldout.accuracy * 100).toFixed(1)}% · digit ${((ckpt.finalHoldout.digitAccuracy ?? 0) * 100).toFixed(1)}% on ${ckpt.finalHoldout.n} exprs`;
      const score = await curriculumScore(model);
      smartnessEl.textContent = `Curriculum ${score.ok}/${score.total} (${((score.ok / score.total) * 100).toFixed(0)}%) — from saved ${ckpt.label}`;
      normsEl.innerHTML = `<p class="norm-intro">Loaded ${ckpt.arch.paramCount.toLocaleString()} parameters from checkpoint (train from scratch to see before→after norm bars).</p>`;
      if (!opts?.silent) {
        log(
          `Loaded checkpoint ${ckpt.id} (${ckpt.arch.paramCount.toLocaleString()} params, ${ckpt.curves.length} epoch curves).`,
        );
      }
      lastRlResult = null;
      postRlCkpt = null;
      rlDeltaEl.textContent = "";
      setRlStatus("");
      evalBtn.disabled = false;
      predictBtn.disabled = false;
      digitRlBtn.disabled = false;
      saveCkptBtn.disabled = true;
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

    const setEvalStatus = (text: string, kind: "" | "ok" | "error" = "") => {
      evalStatusEl.textContent = text;
      evalStatusEl.classList.toggle("is-ok", kind === "ok");
      evalStatusEl.classList.toggle("is-error", kind === "error");
    };

    const runEval = async () => {
      if (!modelReady) {
        setEvalStatus("Load a saved model (or finish Train New) before Run Eval.", "error");
        log("Run Eval: no model loaded yet.");
        return;
      }
      setBusy(true);
      setEvalStatus("Running eval…");
      progressEl.textContent = "Running eval…";
      await paint();
      try {
        let holdSummary = "";
        if (heldoutInput.checked) {
          const held = await evaluateModel(model, batchFromExprs(holdExprs));
          log(
            `eval HOLD       loss=${held.loss.toFixed(4)}  ppl=${formatPpl(held.perplexity)}  exact=${(held.accuracy * 100).toFixed(1)}%  digit=${(held.digitAccuracy * 100).toFixed(1)}%  n=${held.n}`,
          );
          evalLineEl.textContent = `Live held-out: exact ${(held.accuracy * 100).toFixed(1)}% · digit ${(held.digitAccuracy * 100).toFixed(1)}% · ppl ${formatPpl(held.perplexity)} on ${held.n} exprs`;
          holdSummary = `hold-out exact ${(held.accuracy * 100).toFixed(1)}% / digit ${(held.digitAccuracy * 100).toFixed(1)}%`;
          setEvalStatus(`Eval: ${holdSummary}…`, "ok");
          progressEl.textContent = `Eval held-out · running train-dist…`;
          await paint();
        }

        const pipeline = new MathDataPipeline(
          "pretrain",
          new ArithmeticDataset({
            filter: heldoutInput.checked ? (a, b, op) => !isExpressionHeldOut(a, op, b) : undefined,
            mix: NINJA_MIX,
            hardBias: 0.45,
            maxDigits: OPERAND_DIGITS,
          }),
        );
        const { loss, accuracy, digitAccuracy, perplexity } = await trainer.evaluate(model, pipeline, 8);
        log(
          `eval train-dist  loss=${loss.toFixed(4)}  ppl=${formatPpl(perplexity)}  exact=${(accuracy * 100).toFixed(1)}%  digit=${(digitAccuracy * 100).toFixed(1)}%`,
        );
        const trainSummary = `train-dist exact ${(accuracy * 100).toFixed(1)}% / digit ${(digitAccuracy * 100).toFixed(1)}%`;
        setEvalStatus(
          holdSummary ? `Eval done: ${holdSummary} · ${trainSummary}` : `Eval done: ${trainSummary}`,
          "ok",
        );
        progressEl.textContent = `Eval done · ${trainSummary}${holdSummary ? ` · ${holdSummary}` : ""}`;
        evalStatusEl.scrollIntoView({ block: "nearest", behavior: "smooth" });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setEvalStatus(`Eval failed: ${message}`, "error");
        log(`Run Eval failed: ${message}`);
      } finally {
        setBusy(false);
      }
    };

    const runTrain = async () => {
      setBusy(true);
      setEvalStatus("");
      logEl.textContent = "";
      refreshCounts();

      try {
        const epochs = Number(epochsInput.value);
        const batchSize = Math.max(4, Number(batchInput.value) || 64);
        const examples = Math.max(batchSize, Number(examplesInput.value) || 8192);
        const stepsPerEpoch = Math.max(1, Math.round(examples / batchSize));
        const learningRate = Number(lrInput.value);
        const hidden = Math.max(32, Math.min(768, Number(hiddenInput.value) || DEFAULT_HIDDEN));
        hiddenInput.value = String(hidden);
        const useHeldout = heldoutInput.checked;
        const modeLabel = activeMode;
        const preset =
          modeLabel === "custom" ? null : MODE_PRESETS[modeLabel as ModeId];
        const currStart = preset?.curriculumStart ?? 1;
        const currEnd = preset?.curriculumEnd ?? OPERAND_DIGITS;

        const trainFilter = useHeldout
          ? (a: bigint, b: bigint, op: Op) => !isExpressionHeldOut(a, op, b)
          : undefined;
        const useNinjaMix = modeLabel === "ninja" || modeLabel === "mid" || modeLabel === "custom";
        const hardBias = preset?.hardBias ?? 0.45;
        trainDataset = new ArithmeticDataset({
          filter: trainFilter,
          mix: useNinjaMix ? NINJA_MIX : { "+": 0.35, "-": 0.2, "*": 0.3, "/": 0.15 },
          hardBias,
          maxDigits: currStart,
        });
        const pipeline = new MathDataPipeline("pretrain", trainDataset);
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
          `Train New [${modeLabel}] — ${OPERAND_DIGITS}-digit place MLP · H=${hidden} (~${TinyMathMlp.paramCount(hidden).toLocaleString()} params) · curriculum ${currStart}→${currEnd} · ${examples}/epoch · lr=${learningRate}`,
        );
        if (modeLabel === "ninja") {
          log("Ninja nine-digit curriculum can take a while — that’s the strong-math setting.");
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
          heldoutExprs: holdExprs,
          curriculumStartDigits: currStart,
          curriculumEndDigits: currEnd,
          onCurriculumDigits: (d) => trainDataset?.setMaxDigits(d),
          onMetrics: async (m) => {
            history.push(m);
            progressFill.style.width = `${((m.epoch / epochs) * 100).toFixed(1)}%`;
            progressEl.textContent = `Epoch ${m.epoch}/${epochs} · maxDigits=${m.maxDigits} · loss ${m.loss.toFixed(3)} · exact ${(m.accuracy * 100).toFixed(1)}% · digit ${(m.digitAccuracy * 100).toFixed(1)}%`;
            let line = `epoch ${String(m.epoch).padStart(3)}  d=${m.maxDigits}  loss=${m.loss.toFixed(4)}  exact=${(m.accuracy * 100).toFixed(1)}%  digit=${(m.digitAccuracy * 100).toFixed(1)}%`;
            if (m.heldout) {
              line += `  |  HOLD exact=${(m.heldout.accuracy * 100).toFixed(1)}%  digit=${(m.heldout.digitAccuracy * 100).toFixed(1)}%`;
              evalLineEl.textContent = `Held-out: exact ${(m.heldout.accuracy * 100).toFixed(1)}% · digit ${(m.heldout.digitAccuracy * 100).toFixed(1)}% on ${m.heldout.n} exprs`;
            }
            log(line);
            renderCurves(curvesEl, history, {
              title: `${modeLabel} · H=${hidden} · curr ${currStart}→${currEnd}`,
            });
            const live = pipeline.nextBatch(6);
            renderDataFeed(dataFeedEl, live.texts ?? [], `Live train sample @ epoch ${m.epoch} (maxDigits=${m.maxDigits}):`);
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
        smartnessEl.textContent = `Curriculum ${score.ok}/${score.total} (${((score.ok / score.total) * 100).toFixed(0)}%) — compare held-out exact.`;
        progressEl.textContent = `Done [${modeLabel}] · ${epochs} epochs · final exact ${(history.at(-1)?.accuracy ?? 0) * 100 | 0}%`;
        progressFill.style.width = "100%";
        modelReady = true;
        const fb = gpu.cpuFallbackCount - fb0;
        log(`Train New finished.${fb ? ` (CPU matmul fallback ${fb}×)` : ""}`);

        const probes: Array<readonly [bigint, Op, bigint]> = [
          [12n, "+", 34n],
          [47n, "*", 19n],
          [123456789n, "+", 987654321n],
          [999999999n, "-", 1n],
          [111111111n, "*", 2n],
          holdExprs[0] ?? [58n, "+", 67n],
        ];
        for (const [a, op, b] of probes) {
          const truth = computeAnswer(a, op, b);
          const { answer } = await model.predict(encodeFeatures(a, op, b));
          const tag = isExpressionHeldOut(a, op, b) ? "HOLD" : "probe";
          log(`probe [${tag}] ${formatExpr(a, op, b)} → ${answer} (want ${truth})${answer === truth ? " ✓" : " ✗"}`);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        log(`Train New failed: ${message}`);
        setEvalStatus(`Train failed: ${message}`, "error");
        modelReady = history.length > 0;
      } finally {
        setBusy(false);
      }
    };

    const runPredict = async () => {
      if (!modelReady) {
        predictionEl.textContent = "Load a saved model or Train New first.";
        return;
      }
      const a = parseOperand(aInput.value);
      const b = parseOperand(bInput.value);
      const op = (document.querySelector("#op") as HTMLSelectElement).value as Op;
      if (a === null || b === null || !OPS.includes(op)) {
        oodEl.hidden = false;
        oodEl.textContent = "Enter non-negative integers (up to 9 digits each).";
        return;
      }

      if (!inDomain(a, b)) {
        oodEl.hidden = false;
        const offender = isBeyondOperandDigits(a) ? a : b;
        oodEl.textContent = `${formatExpr(a, op, b)} is OOD (need 0–${MAX_OPERAND}). ${offender} needs more than ${OPERAND_DIGITS} digits.`;
        predictionEl.textContent = "";
        topkEl.innerHTML = "";
        return;
      }
      oodEl.hidden = true;
      const truth = computeAnswer(a, op, b);
      const { answer, digits, digitProbs } = await model.predict(encodeFeatures(a, op, b));
      const tag = isExpressionHeldOut(a, op, b) ? " [fair held-out]" : "";
      predictionEl.textContent = `${formatExpr(a, op, b)} → model says ${answer} (truth ${truth})${
        answer === truth ? " ✓" : " ✗"
      }${tag}`;
      renderDigitPlaces(topkEl, digitProbs, digits);
    };

    document.querySelectorAll<HTMLButtonElement>("[data-ckpt]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const key = btn.dataset.ckpt as ModeId;
        applyMode(MODE_PRESETS[key]);
        setEvalStatus("");
        void loadSelected();
      });
    });
    loadCkptBtn.addEventListener("click", () => {
      setEvalStatus("");
      void loadSelected();
    });
    trainBtn.addEventListener("click", () => void runTrain());
    evalBtn.addEventListener("click", () => void runEval());
    predictBtn.addEventListener("click", () => void runPredict());
    aInput.addEventListener("change", () => void runPredict());
    bInput.addEventListener("change", () => void runPredict());

    for (const el of [hiddenInput, batchInput, examplesInput, epochsInput, lrInput]) {
      el.addEventListener("change", () => {
        markCustom();
        refreshCounts();
      });
    }

    const formatPct = (x: number) => `${(x * 100).toFixed(1)}%`;
    const formatDelta = (d: number) => `${d >= 0 ? "+" : ""}${(d * 100).toFixed(1)} pp`;

    const buildPostRlCheckpoint = (result: DigitRlResult): CheckpointFile => {
      const baseId = (loadedCkpt?.id ?? (activeMode === "custom" ? DEFAULT_MODE : activeMode)) as CheckpointId;
      const baseLabel = loadedCkpt?.label ?? MODE_PRESETS[baseId]?.label ?? String(baseId);
      const preset = MODE_PRESETS[baseId];
      const hp = loadedCkpt?.hyperparams ?? {
        hidden: model.hidden,
        epochs: Number(epochsInput.value) || 0,
        lr: Number(lrInput.value) || 0,
        examples: Number(examplesInput.value) || 0,
        batch: Number(batchInput.value) || 64,
        mix: baseId === "simple" ? ("simple" as const) : ("ninja" as const),
        hardBias: preset?.hardBias ?? 0.45,
        heldout: heldoutInput.checked,
        curriculumStart: preset?.curriculumStart ?? 1,
        curriculumEnd: preset?.curriculumEnd ?? OPERAND_DIGITS,
        operandDigits: OPERAND_DIGITS,
        answerDigits: ANSWER_DIGITS,
      };
      const meta = buildCheckpointMeta({
        id: baseId,
        label: `${baseLabel} + ${result.kind}`,
        blurb: `${result.kind} (${result.steps} steps) on ${baseLabel}. Hold-out ${formatPct(result.beforeHoldout.accuracy)} → ${formatPct(result.afterHoldout.accuracy)} (${formatDelta(result.holdoutDelta)}). ${describePostTrain(result.kind)}`,
        hyperparams: hp,
        holdout: {
          pct: FAIR_HOLDOUT_PCT,
          n: result.afterHoldout.n,
          seed: FAIR_HOLDOUT_SEED,
        },
        curves: history,
        finalHoldout: result.afterHoldout,
        probes: loadedCkpt?.probes,
        paramCount: TinyMathMlp.paramCount(model.hidden),
      });
      const ckpt = packCheckpoint(meta, model);
      ckpt.postTrain = {
        kind: result.kind,
        steps: result.steps,
        meanReward: result.meanReward,
        beforeHoldout: result.beforeHoldout,
        afterHoldout: result.afterHoldout,
        holdoutDelta: result.holdoutDelta,
        baseId,
      };
      return ckpt;
    };

    const runDigitRlUi = async () => {
      if (!modelReady) {
        setRlStatus("Load a saved model (or finish Train New) before post-train.", "error");
        return;
      }
      const kind = postKindSel.value as PostTrainKind;
      setBusy(true);
      setRlStatus(`${kind}: measuring hold-out before…`);
      rlDeltaEl.textContent = "";
      postRlCkpt = null;
      lastRlResult = null;
      await paint();
      const steps = Math.max(5, Math.min(400, Number(rlStepsInput.value) || 80));
      const rlLr = Math.max(0.0001, Number(rlLrInput.value) || 0.001);
      rlStepsInput.value = String(steps);
      rlLrInput.value = String(rlLr);
      const beforeNorms = model.weightNorms();
      log(
        `Post-train ${kind} — ${describePostTrain(kind)} · steps=${steps} · lr=${rlLr} · held-out bank n=${holdExprs.length}`,
      );
      try {
        const result = await runPostTrain(model, {
          kind,
          steps,
          learningRate: rlLr,
          holdoutExprs: holdExprs,
          maxDigits: OPERAND_DIGITS,
          onStep: async (m) => {
            progressFill.style.width = `${((m.step / m.steps) * 100).toFixed(1)}%`;
            progressEl.textContent = `${kind} ${m.step}/${m.steps} · reward ${m.meanReward.toFixed(3)} · digit-acc ${(m.meanDigitAcc * 100).toFixed(1)}% · exact ${(m.exactRate * 100).toFixed(1)}%`;
            setRlStatus(
              `${kind} step ${m.step}/${m.steps}: reward ${m.meanReward.toFixed(3)} · greedy digits ${(m.greedyDigitAcc * 100).toFixed(1)}%`,
            );
            if (m.step === m.steps || m.step % Math.max(1, Math.floor(m.steps / 6)) === 0) {
              log(
                `pt ${String(m.step).padStart(3)}  reward=${m.meanReward.toFixed(4)}  digit=${(m.meanDigitAcc * 100).toFixed(1)}%  exact=${(m.exactRate * 100).toFixed(1)}%`,
              );
              await paint();
            }
          },
        });
        lastRlResult = result;
        postRlCkpt = buildPostRlCheckpoint(result);
        renderNormBars(normsEl, beforeNorms, model.weightNorms());
        const before = formatPct(result.beforeHoldout.accuracy);
        const after = formatPct(result.afterHoldout.accuracy);
        const delta = formatDelta(result.holdoutDelta);
        rlDeltaEl.textContent = `Hold-out before ${before} → after ${after} (${delta}) · n=${result.afterHoldout.n}`;
        evalLineEl.textContent = `After ${kind}: hold-out ${after} (was ${before}, ${delta}) on ${result.afterHoldout.n} exprs`;
        setRlStatus(`${kind} done · mean reward ${result.meanReward.toFixed(3)} · ${delta} hold-out`, "ok");
        progressEl.textContent = `${kind} done · hold-out ${before} → ${after}`;
        progressFill.style.width = "100%";
        ckptMetaEl.textContent = `${postRlCkpt.label}: hold-out ${before} → ${after} (${delta}). Hit Save checkpoint to download JSON.`;
        log(
          `${kind} finished: HOLD ${before} → ${after} (${delta}) · mean reward ${result.meanReward.toFixed(4)}`,
        );
        const score = await curriculumScore(model);
        smartnessEl.textContent = `Curriculum ${score.ok}/${score.total} (${((score.ok / score.total) * 100).toFixed(0)}%) — after ${kind}`;
        rlStatusEl.scrollIntoView({ block: "nearest", behavior: "smooth" });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setRlStatus(`Post-train failed: ${message}`, "error");
        log(`Post-train failed: ${message}`);
      } finally {
        setBusy(false);
      }
    };

    const savePostRlCheckpoint = () => {
      if (!postRlCkpt) {
        setRlStatus("Run post-train first, then Save checkpoint.", "error");
        return;
      }
      const fname = `${postRlCkpt.id}-${postRlCkpt.postTrain?.kind ?? "post"}-checkpoint.json`;
      downloadCheckpoint(postRlCkpt, fname);
      setRlStatus(`Saved ${fname} (${postRlCkpt.arch.paramCount.toLocaleString()} params).`, "ok");
      log(`Downloaded ${fname}`);
    };

    digitRlBtn.addEventListener("click", () => void runDigitRlUi());
    saveCkptBtn.addEventListener("click", () => savePostRlCheckpoint());

    const previewFilter = (a: bigint, b: bigint, op: Op) => !isExpressionHeldOut(a, op, b);
    const previewPipe = new MathDataPipeline(
      "pretrain",
      new ArithmeticDataset({
        filter: previewFilter,
        mix: NINJA_MIX,
        hardBias: 0.45,
        maxDigits: 4,
      }),
    );
    renderDataFeed(
      dataFeedEl,
      previewPipe.nextBatch(6).texts ?? [],
      "Preview of train-distribution rows (curriculum sample @ 4 digits; fair held-out excluded):",
    );

    await loadSelected();
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    statusEl.innerHTML = `<p><strong>WebGPU failed.</strong> ${message}</p>`;
    log(message);
  }
}

boot();
