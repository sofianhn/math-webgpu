# How we taught a browser MLP to be a nine-digit calculator

*Place-value features, independent digit heads, staircase curriculum, digit RL / DPO — and the dead ends that looked clever until the probes failed.*

---

The live demo loads a **Ninja** checkpoint and lets you probe expressions up to nine digits, then optionally **post-train** with Digit RL or DPO-digits. This note is the engineering diary: what had to change when we left the cozy two-digit world.

## The problem with “just add more classes”

A flat softmax over answers works for toys (`0…999`). It dies for nine-digit operands:

- Addition needs up to **10** answer digits.
- Multiplication needs up to **18**.
- You cannot materialize a \(10^{18}\)-way classifier in a tab.

So the output is **digit heads**, LSD-first: eighteen independent 10-way classifiers. Exact match means every place is right — including the high zeros that pin the magnitude.

## Representation: nine place-value slots per operand

Each operand is encoded as **nine one-hot place slots** (ones…10⁸), plus a 4-way operator one-hot:

\[
\texttt{inputDim} = 9\cdot10 + 4 + 9\cdot10 = 184
\]

Shorter numbers are zero-padded in the high places. That is what makes curriculum possible: a one-digit `3` and a nine-digit `300000000` share the ones-slot structure.

BigInt answers (products can exceed `Number.MAX_SAFE_INTEGER`) are split with `toDigits` / `fromDigits` so training never silently corrupts large values.

## Architecture: independent heads beat a shared head

We tried a **shared** digit head conditioned on place embeddings. On the old two-digit domain it stalled around **~19%** hold-out exact while digit accuracy looked “fine” (~94%) — classic leading-zero theater plus a head that copied the ones digit upward.

Switching back to **independent per-place heads** (the successful H/T/O idea, extended to 18 places) restored strength: the same two-digit stress test reached **~96%** hold-out exact and probes like `47 × 19` landed again.

Trunk: `184 → H → ReLU → H → ReLU`, then `Wheads[p]` for `p = 0…17`. Hybrid WebGPU path: matmuls on GPU, bias/ReLU/softmax/backward on CPU.

## Loss: supervise every place, weight the zeros

If you only supervise “significant” digits, high places never learn to stay `0`. The shared-head bug was partly this; even with independent heads it matters.

We supervise **all 18 places** every step, with weights:

- significant + near places ≈ `1.15–1.35` (carries emphasized on low places)
- far trailing zeros ≈ `0.18–0.55`

Metrics report **exact** (all places) and **digit** (weighted/active). Exact is the calculator scoreboard.

## Data: rich mix + staircase curriculum + rehearsal

Sampling is not a tiny grid anymore. We draw across digit lengths with:

- **Op mix** biased toward `×` and carry-heavy `+` (Ninja)
- **Hard bias** for carries, borrows, multi-digit products, non-trivial divides
- **Staircase curriculum**: linger on each max-length before advancing `1 → 9`
- **Short-length rehearsal** (~45% uniform over `1…max`) so advancing to 7 digits does not erase `3 + 5`

Fair hold-out is hash-based (~10% of expressions). During training we score a **length-matched** hold-out slice so the curve is not punished for nine-digit probes while the model is still on three digits.

## Pretrain presets

| Mode | Hidden | Curriculum | Role |
| --- | ---: | --- | --- |
| Simple | 256 | 1→4 | Fast sanity, shorter operands |
| Mid | 384 | 1→7 | Compromise |
| Ninja | 512 | 1→9 | Default demo / strong story |

Checkpoints store weights (f32→base64), hyperparams, per-epoch curves, final hold-out, and length-bucket probes.

## Post-training without English

The MLP has no language channel — no chain-of-thought. Post-train still helps:

1. **Digit RL** — sample digits from the policy, reward digit-wise correctness (+ exact bonus), self-critical REINFORCE vs greedy baseline, light CE mix so a good pretrained net is not destroyed.
2. **DPO-digits** — prefer the true digit sequence over a one-place near-miss for the same expression.
3. **Self-consistency** — temperature samples + majority vote at inference (no weight update).

Run them in the browser on a loaded checkpoint; Save downloads a new JSON with before/after hold-out deltas.

## What actually moved the needle

In rough order of impact for this project:

1. **Independent digit heads** (vs shared+emb)
2. **Supervise high zeros** (vs significant-only loss)
3. **Place-value inputs** (vs flat operand IDs)
4. **Staircase curriculum + rehearsal** (vs linear jump that forgets)
5. **Carry/mul-weighted data** (vs uniform ops)
6. **Digit RL / DPO** (polish after a strong pretrain)
7. **WebGPU matmul** (speed, not accuracy)

## Dead ends worth remembering

- Flat answer classes past three digits.
- Shared digit head that looked healthy on mean digit accuracy.
- Curriculum that leaped `1 → 2` without rehearsal — wiped `3 + 5`.
- Judging progress only on train exact while hold-out mixed longer lengths.

## Try it

Live demo: [sofianhn.github.io/math-webgpu](https://sofianhn.github.io/math-webgpu/)

```bash
npm install
npm run dev
# offline:
npm run pretrain:checkpoints
npm run rl:digit
```

The honest ceiling for a tiny feedforward MLP is not “arbitrary precision calculator.” It is **structured digit prediction** with curriculum and post-train. That is still enough to make a browser demo feel like a real arithmetic model — and to teach why representation beats epoch count.
