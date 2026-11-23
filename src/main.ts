import { WebGpuBackend } from "./webgpu";
import { DEFAULT_HIDDEN, TinyMathMlp } from "./model";
import {
  ArithmeticDataset,
  CURRICULUM,
  MAX_OPERAND,
  MathDataPipeline,
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

const statusEl = document.querySelector("#status")!;
const controlsEl = document.querySelector<HTMLElement>("#controls")!;
const dataEl = document.querySelector<HTMLElement>("#data")!;
const dataFeedEl = document.querySelector<HTMLElement>("#data-feed")!;
const holdFeedEl = document.querySelector<HTMLElement>("#hold-feed")!;
const internalsEl = document.querySelector<HTMLElement>("#internals")!;
const logEl = document.querySelector("#log")!;
const probeEl = document.querySelector<HTMLElement>("#probe")!;
const trainBtn = document.querySelector<HTMLButtonElement>("#train")!;
const evalBtn = document.querySelector<HTMLButtonElement>("#eval")!;
const predictBtn = document.querySelector<HTMLButtonElement>("#predict")!;
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
const epochsInput = document.querySelector<HTMLInputElement>("#epochs")!;
const batchInput = document.querySelector<HTMLInputElement>("#batch")!;
const examplesInput = document.querySelector<HTMLInputElement>("#examples")!;
const lrInput = document.querySelector<HTMLInputElement>("#lr")!;
const hiddenInput = document.querySelector<HTMLInputElement>("#hidden")!;
const heldoutInput = document.querySelector<HTMLInputElement>("#heldout")!;
const aInput = document.querySelector<HTMLInputElement>("#a")!;
const bInput = document.querySelector<HTMLInputElement>("#b")!;

const SIZE_PRESETS: Record<string, number> = {
  tiny: 32,
  small: 64,
  medium: 96,
  large: 192,
  xlarge: 256,
};

const DATA_PRESETS: Record<string, number> = {
  thin: 1024,
  default: 2560,
  rich: 5120,
};

const LR_PRESETS: Record<string, { epochs: number; lr: number }> = {
  careful: { epochs: 80, lr: 0.001 },
  default: { epochs: 40, lr: 0.01 },
  spicy: { epochs: 30, lr: 0.08 },
};

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

function refreshCounts() {
  const hidden = Math.max(16, Math.min(512, Number(hiddenInput.value) || DEFAULT_HIDDEN));
  const batch = Math.max(4, Number(batchInput.value) || 64);
  let examples = Math.max(batch, Number(examplesInput.value) || 2560);
  examples = Math.max(batch, Math.round(examples / batch) * batch);
  examplesInput.value = String(examples);
  const steps = examples / batch;
  const tokens = examples * TOKENS_PER_EXAMPLE;
  const params = TinyMathMlp.paramCount(hidden);
  paramCountEl.textContent = `Parameters ≈ ${params.toLocaleString()}  (trunk W1 ${44}×${hidden} + 3 digit heads × (${hidden}×10+10) — no flat 1000-way W2)`;
  tokenCountEl.textContent = `Tokens / epoch ≈ ${tokens.toLocaleString()}  (${examples.toLocaleString()} examples × ${TOKENS_PER_EXAMPLE} tokens · ${steps} steps × batch ${batch})`;
}

function setActive(group: string, name: string) {
  document.querySelectorAll<HTMLButtonElement>(`[data-${group}]`).forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset[group] === name);
  });
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
    statusEl.innerHTML = `<p><strong>WebGPU ready.</strong> Digit-wise answer heads + fair expression holdout. Domain operands <strong>0–${MAX_OPERAND}</strong>.</p>`;
    controlsEl.hidden = false;
    dataEl.hidden = false;
    internalsEl.hidden = false;
    probeEl.hidden = false;
    renderCurves(curvesEl, []);
    refreshCounts();

    const holdExprs = buildExpressionHoldout(200, FAIR_HOLDOUT_PCT);
    renderDataFeed(
      holdFeedEl,
      holdExprs.slice(0, 8).map(([a, op, b]) => {
        const ans = computeAnswer(a, op, b);
        return `${formatExpr(a, op, b, ans)}  ·  fair held-out (frozen expression)`;
      }),
      `Frozen fair held-out bank: ${holdExprs.length} exact expressions (~${FAIR_HOLDOUT_PCT}% slice). All digits 0–9 still appear elsewhere in training.`,
    );

    let model = new TinyMathMlp(gpu, DEFAULT_HIDDEN);
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

    document.querySelectorAll<HTMLButtonElement>("[data-size]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const key = btn.dataset.size!;
        setActive("size", key);
        if (key !== "custom" && SIZE_PRESETS[key] != null) {
          hiddenInput.value = String(SIZE_PRESETS[key]);
        }
        refreshCounts();
      });
    });
    document.querySelectorAll<HTMLButtonElement>("[data-data]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const key = btn.dataset.data!;
        setActive("data", key);
        if (key !== "custom" && DATA_PRESETS[key] != null) {
          examplesInput.value = String(DATA_PRESETS[key]);
        }
        refreshCounts();
      });
    });
    document.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const key = btn.dataset.preset!;
        setActive("preset", key);
        const p = LR_PRESETS[key];
        if (p) {
          epochsInput.value = String(p.epochs);
          lrInput.value = String(p.lr);
        }
      });
    });
    for (const el of [hiddenInput, batchInput, examplesInput]) {
      el.addEventListener("change", () => {
        if (el === hiddenInput) setActive("size", "custom");
        if (el === examplesInput || el === batchInput) setActive("data", "custom");
        refreshCounts();
      });
    }

    const previewFilter = (a: number, b: number, op: Op) => !isExpressionHeldOut(a, op, b);
    const previewPipe = new MathDataPipeline("pretrain", new ArithmeticDataset(previewFilter));
    renderDataFeed(
      dataFeedEl,
      previewPipe.nextBatch(6).texts ?? [],
      "Preview of train-distribution rows (fair held-out expressions excluded when eval is on):",
    );

    trainBtn.addEventListener("click", async () => {
      trainBtn.disabled = true;
      evalBtn.disabled = true;
      predictBtn.disabled = true;
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

      const trainFilter = useHeldout
        ? (a: number, b: number, op: Op) => !isExpressionHeldOut(a, op, b)
        : undefined;
      const pipeline = new MathDataPipeline("pretrain", new ArithmeticDataset(trainFilter));
      const heldoutBatch = useHeldout ? batchFromExprs(holdExprs) : undefined;

      model = new TinyMathMlp(gpu, hidden);
      const beforeNorms = model.weightNorms();
      history = [];
      renderCurves(curvesEl, history);
      progressFill.style.width = "0%";
      progressEl.textContent = `Starting… 0 / ${epochs}`;
      evalLineEl.textContent = "";
      const fb0 = gpu.cpuFallbackCount;

      log(
        `Pretrain — digit heads · H=${hidden} (~${TinyMathMlp.paramCount(hidden).toLocaleString()} params) · ${examples}/epoch · ${stepsPerEpoch} steps × batch ${batchSize} · ~${(examples * TOKENS_PER_EXAMPLE).toLocaleString()} tokens/epoch · lr=${learningRate} · held-out=${useHeldout ? `fair ~${FAIR_HOLDOUT_PCT}% exprs` : "off"}`,
      );
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
            evalLineEl.textContent = `Held-out (unseen exprs): perplexity ${formatPpl(m.heldout.perplexity)} · acc ${(m.heldout.accuracy * 100).toFixed(1)}% on ${m.heldout.n} exprs (~${m.heldout.tokens} tokens)`;
          }
          log(line);
          renderCurves(curvesEl, history, { title: `digit-MLP H=${hidden} · ${examples} ex/ep · lr=${learningRate}` });
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
      smartnessEl.textContent = `Curriculum ${score.ok}/${score.total} (${((score.ok / score.total) * 100).toFixed(0)}%) — digit heads + fair holdout: rising held-out acc means real generalization.`;
      progressEl.textContent = `Done · ${epochs} epochs · final train ppl ${formatPpl(history.at(-1)?.perplexity ?? NaN)}`;
      progressFill.style.width = "100%";
      const fb = gpu.cpuFallbackCount - fb0;
      log(`Pretrain finished.${fb ? ` (CPU matmul fallback ${fb}×)` : ""}`);

      // Prefer a held-out expression from the bank if available.
      const holdProbe = holdExprs.find(([a, op, b]) => op === "+" && computeAnswer(a, op, b) < 1000) ?? holdExprs[0];
      const probes: Array<readonly [number, Op, number]> = [
        [12, "+", 34],
        holdProbe ?? [17, "+", 2],
        [15, "*", 6],
        [50, "-", 17],
      ];
      for (const [a, op, b] of probes) {
        const truth = computeAnswer(a, op, b);
        const { answer } = await model.predict(encodeFeatures(a, op, b));
        const tag = isExpressionHeldOut(a, op, b) ? "HOLD" : "train-dist";
        log(`probe [${tag}] ${formatExpr(a, op, b)} → ${answer} (want ${truth})${answer === truth ? " ✓" : " ✗"}`);
      }

      trainBtn.disabled = false;
      evalBtn.disabled = false;
      predictBtn.disabled = false;
    });

    evalBtn.addEventListener("click", async () => {
      evalBtn.disabled = true;
      const pipeline = new MathDataPipeline(
        "pretrain",
        new ArithmeticDataset(
          heldoutInput.checked ? (a, b, op) => !isExpressionHeldOut(a, op, b) : undefined,
        ),
      );
      const { loss, accuracy, perplexity } = await trainer.evaluate(model, pipeline, 15);
      log(`eval train-dist  loss=${loss.toFixed(4)}  ppl=${formatPpl(perplexity)}  acc=${(accuracy * 100).toFixed(1)}%`);
      if (heldoutInput.checked) {
        const held = await evaluateModel(model, batchFromExprs(holdExprs));
        log(
          `eval HOLD       loss=${held.loss.toFixed(4)}  ppl=${formatPpl(held.perplexity)}  acc=${(held.accuracy * 100).toFixed(1)}%  n=${held.n}`,
        );
      }
      evalBtn.disabled = false;
    });

    const runPredict = async () => {
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
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    statusEl.innerHTML = `<p><strong>WebGPU failed.</strong> ${message}</p>`;
    log(message);
  }
}

boot();
