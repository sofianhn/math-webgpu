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

    const model = new TinyMathMlp(gpu);
    const trainer = new MathTrainer();
    const pipeline = new MathDataPipeline("pretrain");

    // Smoke-test one GPU matmul before training UI is used.
    const smoke = await gpu.matmul(
      new Float32Array([1, 2, 3, 4]),
      new Float32Array([5, 6, 7, 8]),
      2,
      2,
      2,
    );
    const expected = [19, 22, 43, 50];
    const ok = expected.every((v, i) => Math.abs(smoke[i] - v) < 1e-3);
    log(ok ? "GPU matmul smoke-test passed (2×2)." : `GPU matmul mismatch: ${Array.from(smoke)}`);

    trainBtn.addEventListener("click", async () => {
      trainBtn.disabled = true;
      evalBtn.disabled = true;
      logEl.textContent = "";
      const epochs = Number((document.querySelector("#epochs") as HTMLInputElement).value);
      const batchSize = Number((document.querySelector("#batch") as HTMLInputElement).value);
      const learningRate = Number((document.querySelector("#lr") as HTMLInputElement).value);
      log(`Pretrain start — epochs=${epochs} batch=${batchSize} lr=${learningRate}`);

      await trainer.train(model, pipeline, {
        stage: "pretrain",
        epochs,
        batchSize,
        learningRate,
        onMetrics: (m) =>
          log(`epoch ${m.epoch.toString().padStart(3)}  loss=${m.loss.toFixed(4)}  acc=${(m.accuracy * 100).toFixed(1)}%`),
      });

      log("Pretrain finished.");
      trainBtn.disabled = false;
      evalBtn.disabled = false;
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
