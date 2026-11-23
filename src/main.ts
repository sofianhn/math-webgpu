import { WebGpuBackend } from "./webgpu";
import { DEFAULT_HIDDEN, TinyMathMlp } from "./model";
import {
  CURRICULUM,
  MAX_OPERAND,
  MathDataPipeline,
  OPS,
  computeAnswer,
  encodeFeatures,
  formatExpr,
  inDomain,
  type Op,
} from "./data";
import { MathTrainer } from "./trainer";
import type { TrainMetrics } from "./types";
import { renderCurves, renderNormBars, renderTopK } from "./viz";

const statusEl = document.querySelector("#status")!;
const controlsEl = document.querySelector<HTMLElement>("#controls")!;
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
const epochsInput = document.querySelector<HTMLInputElement>("#epochs")!;
const batchInput = document.querySelector<HTMLInputElement>("#batch")!;
const lrInput = document.querySelector<HTMLInputElement>("#lr")!;
const hiddenInput = document.querySelector<HTMLInputElement>("#hidden")!;
const aInput = document.querySelector<HTMLInputElement>("#a")!;
const bInput = document.querySelector<HTMLInputElement>("#b")!;

const PRESETS: Record<string, { epochs: number; batch: number; lr: number; hidden: number }> = {
  careful: { epochs: 80, batch: 32, lr: 0.001, hidden: 96 },
  default: { epochs: 40, batch: 64, lr: 0.01, hidden: 96 },
  spicy: { epochs: 25, batch: 64, lr: 0.08, hidden: 64 },
};

function log(line: string) {
  logEl.textContent += line + "\n";
  logEl.scrollTop = logEl.scrollHeight;
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

function applyPreset(name: string) {
  const p = PRESETS[name];
  if (!p) return;
  epochsInput.value = String(p.epochs);
  batchInput.value = String(p.batch);
  lrInput.value = String(p.lr);
  hiddenInput.value = String(p.hidden);
  document.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.preset === name);
  });
}

async function boot() {
  try {
    const gpu = await WebGpuBackend.create();
    statusEl.innerHTML = `<p><strong>WebGPU ready.</strong> Forward matmuls use the compute shader (with CPU fallback if the driver disagrees).</p>`;
    controlsEl.hidden = false;
    internalsEl.hidden = false;
    probeEl.hidden = false;
    renderCurves(curvesEl, []);

    let model = new TinyMathMlp(gpu, DEFAULT_HIDDEN);
    const trainer = new MathTrainer();
    const pipeline = new MathDataPipeline("pretrain");
    let history: TrainMetrics[] = [];

    const smoke = await gpu.matmul(
      new Float32Array([1, 2, 3, 4]),
      new Float32Array([5, 6, 7, 8]),
      2,
      2,
      2,
    );
    const expected = [19, 22, 43, 50];
    const ok2 = expected.every((v, i) => Math.abs(smoke[i] - v) < 1e-3);
    log(ok2 ? "GPU matmul smoke-test passed (2×2)." : `GPU matmul mismatch: ${Array.from(smoke)}`);

    document.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach((btn) => {
      btn.addEventListener("click", () => applyPreset(btn.dataset.preset!));
    });

    trainBtn.addEventListener("click", async () => {
      trainBtn.disabled = true;
      evalBtn.disabled = true;
      predictBtn.disabled = true;
      logEl.textContent = "";
      const epochs = Number(epochsInput.value);
      const batchSize = Number(batchInput.value);
      const learningRate = Number(lrInput.value);
      const hidden = Math.max(16, Math.min(256, Number(hiddenInput.value) || DEFAULT_HIDDEN));
      hiddenInput.value = String(hidden);

      model = new TinyMathMlp(gpu, hidden);
      const beforeNorms = model.weightNorms();
      history = [];
      renderCurves(curvesEl, history);
      const fb0 = gpu.cpuFallbackCount;
      log(
        `Pretrain start — epochs=${epochs} batch=${batchSize} lr=${learningRate} hidden=${hidden} · domain digits 0–${MAX_OPERAND}`,
      );
      log(`Why more epochs won’t fix 3+50: 50 was never encoded in training.`);

      await trainer.train(model, pipeline, {
        stage: "pretrain",
        epochs,
        batchSize,
        learningRate,
        onMetrics: (m) => {
          history.push(m);
          if (m.epoch === 1 || m.epoch === epochs || m.epoch % Math.max(1, Math.floor(epochs / 10)) === 0) {
            log(
              `epoch ${m.epoch.toString().padStart(3)}  loss=${m.loss.toFixed(4)}  acc=${(m.accuracy * 100).toFixed(1)}%`,
            );
          }
          renderCurves(curvesEl, history, {
            title: `lr=${learningRate} · batch=${batchSize} · hidden=${hidden}`,
          });
        },
      });

      const afterNorms = model.weightNorms();
      renderNormBars(normsEl, beforeNorms, afterNorms);

      const score = await curriculumScore(model);
      const pct = ((score.ok / score.total) * 100).toFixed(0);
      smartnessEl.textContent = `Smartness (curriculum ${score.ok}/${score.total}): ${pct}% — try another preset and retrain to see how knobs change this.`;

      const fb = gpu.cpuFallbackCount - fb0;
      log(`Pretrain finished.${fb ? ` (CPU matmul fallback ${fb}×)` : ""}`);
      for (const [a, op, b] of CURRICULUM.slice(0, 4)) {
        const truth = computeAnswer(a, op, b);
        const { answer } = await model.predict(encodeFeatures(a, op, b));
        log(`probe ${formatExpr(a, op, b)} → ${answer} (want ${truth})${answer === truth ? " ✓" : " ✗"}`);
      }

      trainBtn.disabled = false;
      evalBtn.disabled = false;
      predictBtn.disabled = false;
    });

    evalBtn.addEventListener("click", async () => {
      evalBtn.disabled = true;
      const { loss, accuracy } = await trainer.evaluate(model, pipeline, 15);
      log(`eval  loss=${loss.toFixed(4)}  acc=${(accuracy * 100).toFixed(1)}%`);
      evalBtn.disabled = false;
    });

    const runPredict = async () => {
      const a = Number(aInput.value);
      const b = Number(bInput.value);
      const op = (document.querySelector("#op") as HTMLSelectElement).value as Op;
      if (!OPS.includes(op)) return;

      if (!inDomain(a, b)) {
        oodEl.hidden = false;
        oodEl.textContent =
          `${formatExpr(a, op, b)} is outside the trained domain (digits 0–${MAX_OPERAND} only). ` +
          `One-hot features for out-of-range digits stay empty, so the model is guessing. ` +
          `More epochs will not help — it never saw digit ${a > MAX_OPERAND ? a : b} during training.`;
        predictionEl.textContent = "";
        topkEl.innerHTML = "";
        return;
      }

      oodEl.hidden = true;
      const truth = computeAnswer(a, op, b);
      const { answer, probs } = await model.predict(encodeFeatures(a, op, b));
      predictionEl.textContent = `${formatExpr(a, op, b)} → model says ${answer} (truth ${truth})${
        answer === truth ? " ✓" : " ✗"
      }`;
      renderTopK(topkEl, probs, 5);
    };

    predictBtn.addEventListener("click", () => void runPredict());
    aInput.addEventListener("change", () => {
      if (Number(aInput.value) > MAX_OPERAND || Number(aInput.value) < 0) void runPredict();
    });
    bInput.addEventListener("change", () => {
      if (Number(bInput.value) > MAX_OPERAND || Number(bInput.value) < 0) void runPredict();
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    statusEl.innerHTML = `<p><strong>WebGPU failed.</strong> ${message}</p>`;
    log(message);
  }
}

boot();
