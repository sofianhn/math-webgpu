# Math WebGPU

Browser TypeScript + WebGPU: a tiny MLP learns to act like a **nine-digit calculator**
(add / sub / mul / div on operands up to \(10^9-1\)).

Pipeline: **pretrain → digit RL / DPO-digits → probe**. Checkpoints ship with the Pages build.

## Live demo

**https://sofianhn.github.io/math-webgpu/**

Techniques write-up: [docs/training-the-tiny-math-mlp.md](./docs/training-the-tiny-math-mlp.md) · [live blog](https://sofianhn.github.io/math-webgpu/training.html)

## Requirements

- Node.js 18+
- A WebGPU browser (Chrome/Edge 113+, Safari 17+, or Firefox with `dom.webgpu.enabled`)
- Serve over localhost or HTTPS

## Install & run

```bash
npm install
npm run dev
```

Offline pretrain / post-train:

```bash
npm run pretrain:checkpoints          # Simple / Mid / Ninja → public/checkpoints/
FAST=1 npm run pretrain:checkpoints   # short smoke
CHECKPOINT_ONLY=ninja npm run pretrain:checkpoints
npm run rl:digit                      # Digit RL on saved Ninja
KIND=dpo-digits WRITE=1 npm run rl:digit
```

## Layout

```
math-webgpu/
  index.html
  docs/training-the-tiny-math-mlp.md
  public/checkpoints/     # v2 nine-digit weights + curves
  public/training.html    # techniques blog
  src/
    data.ts               # 9-digit place encoding, curriculum sampler
    model.ts              # trunk + 18 independent digit heads
    trainer.ts            # Adam + staircase curriculum
    posttrain.ts          # digit RL, DPO-digits, self-consistency
    checkpoint.ts         # f32-b64 save/load (version 2)
    main.ts               # WebGPU UI
```

## Hybrid WebGPU path

| Step | Where |
| --- | --- |
| Matmul (×2) | WebGPU compute shader |
| Bias, ReLU, digit heads, softmax / CE / Adam | CPU |

## Stages

- `pretrain` — synthetic arithmetic → digit CE
- `rl` — Digit RL / DPO-digits via `posttrain.ts`
- `finetune` — plug-in point for instruction packs

## Results (v2 nine-digit checkpoints)

| Model | Curriculum | Hold-out exact | After Digit RL |
| --- | --- | ---: | ---: |
| Simple | 1→4 | 82.4% | **90.6%** |
| Mid | 1→7 | 94.5% | **97.7%** |
| Ninja | 1→9 (from Mid) | 92.6% | **95.3%** |

Live: https://sofianhn.github.io/math-webgpu/ · Blog: https://sofianhn.github.io/math-webgpu/training.html
