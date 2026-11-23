# Math WebGPU

Minimal TypeScript + WebGPU project: a tiny MLP learns basic arithmetic (add / sub / mul / div on digits 0–9) in the browser.

Architecture is staged for **pretrain → finetune → rl**. Only **pretrain** (synthetic arithmetic → answer classification) is implemented; interfaces leave clear plug-in points for the later stages.

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

Open the URL Vite prints (default `http://localhost:5173`). Click **Train (pretrain)**, watch loss/accuracy, then **Evaluate** or try an expression.

Clone:

```bash
git clone https://github.com/sofianhn/math-webgpu.git
cd math-webgpu
```

## Layout

```
math-webgpu/
  index.html          Demo UI
  package.json
  tsconfig.json
  vite.config.ts
  README.md
  src/
    main.ts           Boot WebGPU, wire train / eval / predict
    style.css
    types.ts          Model / Dataset / Trainer / TrainingStage interfaces
    tensor.ts         CPU tensor helpers (softmax, CE, ReLU)
    webgpu.ts         navigator.gpu init + matmul compute shader
    data.ts           Synthetic arithmetic Dataset + DataPipeline
    model.ts          TinyMathMlp (GPU forward GEMM, CPU backward)
    trainer.ts        Training loop with stage strategy hook
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
