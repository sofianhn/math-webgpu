import { WebGpuBackend } from "./webgpu";
import { TinyMathMlp } from "./model";
import { MathDataPipeline, OPS, encodeFeatures, formatExpr, type Op } from "./data";
import { MathTrainer } from "./trainer";

const statusEl = document.querySelector("#status")!;
const controlsEl = document.querySelector<HTMLElement>("#controls")!;
const logEl = document.querySelector("#log")!;
const probeEl = document.querySelector<HTMLElement>("#probe")!;
const trainBtn = document.querySelector<HTMLButtonElement>("#train")!;
const evalBtn = document.querySelector<HTMLButtonElement>("#eval")!;
const predictBtn = document.querySelector<HTMLButtonElement>("#predict")!;
const predictionEl = document.querySelector("#prediction")!;

function log(line: string) {
  logEl.textContent += line + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

async function boot() {
  try {
    const gpu = await WebGpuBackend.create();
    statusEl.innerHTML = `<p><strong>WebGPU ready.</strong> Compute matmul kernel loaded.</p>`;
    controlsEl.hidden = false;
    probeEl.hidden = false;

    let model = new TinyMathMlp(gpu);
    const trainer = new MathTrainer();
    const pipeline = new MathDataPipeline("pretrain");

    // Smoke-test: tiny + train-shaped GEMMs (catches driver bugs the 2×2 alone misses).
    const smoke = await gpu.matmul(
      new Float32Array([1, 2, 3, 4]),
      new Float32Array([5, 6, 7, 8]),
      2,
      2,
      2,
    );
    const expected = [19, 22, 43, 50];
    const ok2 = expected.every((v, i) => Math.abs(smoke[i] - v) < 1e-3);
    const A = Float32Array.from({ length: 64 * 24 }, (_, i) => ((i * 17) % 100) / 50 - 1);
    const W = Float32Array.from({ length: 24 * 96 }, (_, i) => ((i * 13) % 100) / 100 - 0.5);
    const beforeFb = gpu.cpuFallbackCount;
    await gpu.matmul(A, W, 64, 24, 96);
    const usedFallback = gpu.cpuFallbackCount > beforeFb;
    log(
      ok2
        ? `GPU matmul smoke-test passed (2×2)${usedFallback ? "; larger GEMM uses CPU fallback (driver mismatch)." : "."}`
        : `GPU matmul mismatch: ${Array.from(smoke)}`,
    );

    trainBtn.addEventListener("click", async () => {
      trainBtn.disabled = true;
      evalBtn.disabled = true;
      predictBtn.disabled = true;
      logEl.textContent = "";
      const epochs = Number((document.querySelector("#epochs") as HTMLInputElement).value);
      const batchSize = Number((document.querySelector("#batch") as HTMLInputElement).value);
      const learningRate = Number((document.querySelector("#lr") as HTMLInputElement).value);
      // Fresh weights each run so a prior broken train cannot linger.
      model = new TinyMathMlp(gpu);
      const fb0 = gpu.cpuFallbackCount;
      log(`Pretrain start — epochs=${epochs} batch=${batchSize} lr=${learningRate}`);

      await trainer.train(model, pipeline, {
        stage: "pretrain",
        epochs,
        batchSize,
        learningRate,
        onMetrics: (m) =>
          log(`epoch ${m.epoch.toString().padStart(3)}  loss=${m.loss.toFixed(4)}  acc=${(m.accuracy * 100).toFixed(1)}%`),
      });

      const fb = gpu.cpuFallbackCount - fb0;
      log(`Pretrain finished.${fb ? ` (CPU matmul fallback used ${fb}× — GPU result disagreed.)` : ""}`);
      // Quick held-out probes so a "loss≈0 but wrong answers" bug is obvious in the log.
      for (const [a, op, b] of [
        [3, "+", 5],
        [2, "*", 4],
        [9, "-", 3],
        [8, "/", 2],
      ] as const) {
        const truth =
          op === "+" ? a + b : op === "*" ? a * b : op === "-" ? Math.max(0, a - b) : Math.floor(a / b);
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

    predictBtn.addEventListener("click", async () => {
      const a = Number((document.querySelector("#a") as HTMLInputElement).value);
      const b = Number((document.querySelector("#b") as HTMLInputElement).value);
      const op = (document.querySelector("#op") as HTMLSelectElement).value as Op;
      if (!OPS.includes(op)) return;
      const { answer } = await model.predict(encodeFeatures(a, op, b));
      predictionEl.textContent = `${formatExpr(a, op, b)} → model says ${answer}`;
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    statusEl.innerHTML = `<p><strong>WebGPU failed.</strong> ${message}</p>`;
    log(message);
  }
}

boot();
