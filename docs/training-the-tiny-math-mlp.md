# Teaching a browser to do arithmetic (badly, then better)

*A walk through the Math WebGPU pretrain loop — what the model sees, what the knobs do, why `3 + 50` used to look cursed, and when a net would learn arithmetic “for real.”*

---

You hit **Train**, watch loss fall, then try an expression. Charts help. But two questions keep coming up:

1. What are those **parameter** bars counting?
2. What **data** is this thing actually training on?

And after we moved the default domain to **two-digit** numbers (0–99): at what point would the model learn arithmetic *intrinsically*, instead of memorizing pairs it saw?

## What data it trains on

Each training row is a random expression:

- operands **a, b ∈ {0…99}** (two digits)
- operator in `{+, −, ×, ÷}`
- target = the integer answer (products that would exceed 999 are resampled so they fit the classifier)

The live **data feed** on the demo shows rows like:

`12 + 34 = 46 · features a=1|2 op=+ b=3|4 → class 46`

So you can see both the human expression and the feature breakdown.

## How numbers are encoded (place value, not a flat ID)

We do **not** give the net a single one-hot over 100 values for `a`. We split each operand into **tens digit** and **ones digit** (10 + 10 slots), plus 4 operator slots, for both operands → **44 input features**.

That means `23` and `27` share the “tens = 2” feature. That is a tiny step toward structure — still a far cry from an addition algorithm.

Answers are **classes 0…999**. The last layer is a giant “which integer?” softmax, not a calculator.

## What the parameter tables are

The knobs (learning rate, epochs, batch, hidden size) are **hyperparameters**: they steer *how* optimization runs.

The bars labeled **W1 / b1 / W2 / b2** are the **model parameters** — the actual arrays of numbers being updated:

| Table | Count (example, hidden=96) | Meaning |
| --- | --- | --- |
| **W1** | 44 × 96 | connections from features → hidden units |
| **b1** | 96 | bias per hidden unit |
| **Wh / Wt / Wo** | 96 × 10 each | hundreds / tens / ones digit heads |
| **bh / bt / bo** | 10 each | digit biases |

(Older builds used a flat **W2** 96×1000 class head — easier to memorize, harder to generalize.)

The bar length is the table’s **L2 norm** (overall magnitude) before → after a train run. Δ tells you how hard that table got shoved. Count tells you how many knobs live in that table. Together they are the network’s **memory**, not the training schedule.

## Why `100 + 3` (three digits) will not be fixed by more epochs

The encoder only has tens and ones places. A value like `100` needs a **hundreds** place that does not exist. Features for that operand stay empty. Extra epochs only revisit 0–99. Out-of-distribution failures are a **representation** problem.

(When the demo was single-digit only, `3 + 50` failed for the same reason — `50` had no slot. Two-digit is now in-domain; three-digit is the new OOD wall.)

## At what point would it learn intrinsically?

**Short answer:** not with this setup alone — even at 100% train accuracy.

This model is trained as **input pattern → answer class**. With enough capacity and epochs it can memorize many (a, op, b) → answer maps. Place-value features may help it reuse digit patterns, but the loss never asks it to *execute* addition. It asks it to *name the right class*.

You start to get something closer to “intrinsic” arithmetic when at least one of these is true:

1. **Representation** forces composition — e.g. digit-by-digit outputs, binary encodings, or continuous magnitudes with relational structure — *and* you evaluate on numbers / lengths never seen in training.
2. **Supervision** teaches a procedure — scratchpads, chain-of-thought, or tool use (“call an adder”), not only the final integer.
3. **Architecture** can run algorithms — RNNs/Transformers over digit sequences, program induction, or neurosymbolic stacks — then you measure **length / digit generalization**.

A useful classroom test: hold out all expressions that contain digit `7` in the ones place, train on the rest, then probe `17 + 2`. If accuracy collapses, you memorized surface patterns. If it holds up *and* digit-level errors look like carry bugs, you may be seeing proto-rules. This demo’s classifier still will not invent a hundreds place for `100`.

## What the knobs change (still true)

| Knob | Feel | Reality |
| --- | --- | --- |
| Learning rate | Careful vs spicy | Adam step size |
| Batch size | Steady vs noisy | Gradient averaging |
| Epochs | Train longer | More passes over 0–99; useless for ≥100 |
| Hidden units | Capacity | Wider layers fit more of the table faster |

## Making it learn math (not just memorize)

Two architecture changes matter more than “make H bigger”:

1. **Digit-wise answer head** — predict hundreds / tens / ones, compose `100·H + 10·T + O`.
2. **Fair expression holdout** — freeze ~10% of exact expressions; keep every digit 0–9 in train.

### Simple vs Ninja (the defaults that matter)

Hardened after `47×19` landed as #2 in top-k (813 vs 893): **more epochs alone is weak** when the net is already close. Levers that mattered:

1. **Data mix** — overweight `×` and carry-heavy `+`
2. **Capacity** — two-layer trunk + H288
3. **Objective** — digit CE with tens weighted 1.35 (tens was the miss)
4. **Epochs** — only a modest bump (150→160) after mix/depth helped

| Mode | Hidden | Epochs | LR | Examples/epoch | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| **Simple** | 128 | 60 | 0.01 | 4096 | Faster, lighter mul mix |
| **Ninja** (default) | 288 | 160 | 0.0045 | 12288 | `47×19` ✓ in `sweep-mul.ts` (~92% hard-mul pack) |

## Saved checkpoints (instant load)

Pretrained **Simple / Mid / Ninja** weights ship under `public/checkpoints/`. Each file stores:

- weights (Float32 → base64)
- hyperparams used for that run
- per-epoch train + **frozen held-out** curves
- final hold-out metrics (+ easy/hard/curriculum probes)

On the live page: pick a saved model → curves + knobs appear immediately → **Run Eval** or **Train New** (from scratch with those knobs). Training from scratch is slow; loading is the fast path.

| Mode | Hidden | Epochs | Role |
| --- | ---: | ---: | --- |
| **Simple** | 128 | 60 | Fast baseline |
| **Mid** | 192 | 100 | Middle ground |
| **Ninja** (default) | 288 | 160 | Hard mul/carry |

## Post-training without English

This MLP has **no language channel** — it cannot learn verbal chain-of-thought or English scratchpads. Useful post-train ideas that still fit:

1. **Digit RL / correctness reward** — sample an answer, reward exact match (or partial credit on hundreds/tens/ones).
2. **DPO-style preference on answer digits** — preferred vs rejected integer (or digit triple) for the same expression.
3. **Self-consistency** — temperature / stochastic digit heads; keep the majority vote at inference.

A stub lives in `src/posttrain.ts` (`runPostTrain`) mapped toward the existing `rl` stage — not wired into Train yet on purpose.

## What to try on the live page

1. Leave **Ninja** selected (auto-loads saved weights), inspect the held-out curve, hit **Run Eval**.
2. Probe `47 + 38`, `17 × 19`, `25 × 25`, plus a frozen held-out row.
3. Switch to **Simple** or **Mid** for other saved runs.
4. **Train New** only if you want to watch a from-scratch run (slow).
5. Try `100 + 3` (OOD) — still no hundreds *input* place.

**A model only learns the world you can represent — and only the objective you optimize.**
