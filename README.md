# Math WebGPU

Minimal TypeScript + WebGPU project: a tiny MLP learns basic arithmetic (add / sub / mul / div on digits 0–9) in the browser.

Architecture is staged for **pretrain → finetune → rl**. Only **pretrain** (synthetic arithmetic → answer classification) is implemented; interfaces leave clear plug-in points for the later stages.

## Live demo (GitHub Pages)

**https://sofianhn.github.io/math-webgpu/**

Two-digit operands **0–99**. Load pretrained **Simple / Mid / Ninja** checkpoints for instant curves + eval, or **Train New** from scratch. See the training blog: [docs/training-the-tiny-math-mlp.md](./docs/training-the-tiny-math-mlp.md).

## Requirements

- **Node.js** 18+ (for Vite)
- A **WebGPU** browser:
  - Chrome or Edge 113+
  - Safari 17+ (macOS / iOS)
  - Firefox: enable `dom.webgpu.enabled` (experimental)
- Serve over **localhost** or HTTPS (`navigator.gpu` is unavailable on insecure remote origins)

## Install & run

```bash
cd math-webgpu
npm install
npm run dev
```

Open the URL Vite prints (default `http://localhost:5173`). A saved **Ninja** checkpoint loads by default — inspect curves, **Run Eval**, or **Train New**. Offline pretrain:

```bash
npm run pretrain:checkpoints
```

Clone:

```bash
git clone https://github.com/sofianhn/math-webgpu.git
cd math-webgpu
```


## Layout

```
math-webgpu/
  index.html          Demo UI (saved models, curves, OOD probe)
  public/checkpoints/ Pretrained Simple / Mid / Ninja (+ curves)
  docs/training-the-tiny-math-mlp.md
  scripts/pretrain-checkpoints.ts
  package.json
  tsconfig.json
  vite.config.ts
  README.md
  src/
    main.ts           Boot WebGPU, load checkpoints / train / eval / predict
    checkpoint.ts     Weights + hyperparams + curve format
    posttrain.ts      Stub: digit RL / DPO digits / self-consistency
    viz.ts            Loss/acc curves, weight norms, top-k probs
    style.css
    presets.ts        Simple / Mid / Ninja hyperparams
    types.ts          Model / Dataset / Trainer / TrainingStage interfaces
    tensor.ts         CPU tensor helpers (softmax, CE, ReLU, matmul)
    webgpu.ts         navigator.gpu init + matmul compute shader
    data.ts           Synthetic arithmetic Dataset + DataPipeline
    model.ts          TinyMathMlp (GPU forward GEMM, CPU backward)
    trainer.ts        Training loop with stage strategy hook
    evals.ts          Frozen held-out bank + evaluateModel
```

## Hybrid WebGPU path

| Step        | Where                                      |
| ----------- | ------------------------------------------ |
| Matmul (×2) | **WebGPU** compute shader                  |
| Bias, ReLU  | CPU                                        |
| Softmax / CE / backward / Adam | CPU                     |

**Extension path:** keep activations and weights in GPU buffers, add WGSL kernels for bias-add / ReLU / softmax / `Xᵀ@dY` GEMMs, and run the full backward on-device. `WebGpuBackend.matmul` is the first kernel to build on.

## Stages

- `pretrain` — implemented (`ArithmeticDataset` → CE classification)
- `finetune` — `MathDataPipeline` / `MathTrainer` reject until you add instruction data + strategy
- `rl` — same; plug preference/reward batches into `DataPipeline` + `StageStrategy`
