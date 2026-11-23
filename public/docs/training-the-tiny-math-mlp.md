# Teaching a browser to do arithmetic (badly, then better)

*A walk through the Math WebGPU pretrain loop — what the model sees, what the knobs do, and why `3 + 50 → 6` is not a bug in the optimizer.*

---

You open the page, hit **Train**, and watch loss fall. By epoch forty it looks almost smug. Then you type `3 + 50` and the model says something absurd like `6`.

That moment is the point of this demo.

## The curriculum is tiny on purpose

The network never sees “numbers” as integers. Each expression is a bag of **one-hot features**:

- digit `a` in `0…9` (10 slots)
- operator `+ − × ÷` (4 slots)
- digit `b` in `0…9` (10 slots)

That is a 24-dimensional input. The answer is a **class** from `0…81` (enough room for `9 × 9`). Training draws random expressions from that closed world — four hundred unique inputs in total. A two-layer MLP with a few dozen hidden units can memorize the whole table.

So when loss hits ~0 and accuracy hits ~100%, the model is not “understanding arithmetic.” It is fitting a lookup table that happens to be arithmetic-shaped.

## Why more epochs will not fix `3 + 50`

Digit `50` has **no feature slot**. Writing `x[50] = 1` into a length-24 vector is silently ignored in JavaScript typed arrays. The `b` one-hot stays all zeros. The network receives “three, plus, ???” — a pattern it never trained on.

Extra epochs only revisit digits `0–9`. They cannot invent a column that does not exist. Out-of-distribution (OOD) failures are a **domain** problem, not a **learning-rate** problem.

That is the first educational lesson: *optimization success on the training distribution does not transfer outside the feature scheme you chose.*

## What the knobs actually change

| Knob | What it feels like | What it really does |
| --- | --- | --- |
| **Learning rate** | Careful vs spicy presets | Step size for Adam. Too low → slow curves. Too high → jagged loss, brittle probes. |
| **Batch size** | Steady vs noisy | Averaging of gradients. Smaller batches → noisier accuracy traces. |
| **Epochs** | “Train longer” | More passes over the same 0–9 table. Helps until the curve flats; useless for OOD. |
| **Hidden units** | Model capacity | Wider layers fit the table faster, but still cannot see digit 50. |

The UI’s **smartness** score is curriculum accuracy on a fixed probe set (`3+5`, `2×4`, …). Retrain with **Careful**, **Default**, and **Spicy** and watch that percentage and the loss curve change. Same domain, different dynamics — that is “parameters affect smartness” in the only sense that matters here.

## Training internals worth staring at

**Loss & accuracy curves.** Loss should drop; accuracy should rise. If loss is flat near zero but probes fail, something else is wrong (we hit that earlier with flaky GPU matmuls — the app now cross-checks GEMMs against a CPU reference).

**Weight L2 norms.** Bars compare parameter magnitudes before and after a run. You are watching the tables `W1`, `b1`, `W2`, `b2` move. A spicy learning rate often shoves norms farther; a careful run moves them gently.

**Top class probabilities.** On a probe, the softmax over answer classes is a confidence fingerprint. In-domain correct answers usually show a sharp peak; OOD inputs look diffuse or arbitrarily peaked on a wrong class.

## The hybrid WebGPU path (one paragraph)

Forward matmuls (`X @ W1`, `H @ W2`) run as a WGSL compute shader when the driver agrees with a CPU reference; bias, ReLU, softmax, cross-entropy, and backward stay on the CPU. That split keeps the demo honest: you still touch a real GPU kernel, but training remains correct on phones with eccentric WebGPU implementations.

## What to try next (as a reader)

1. Train with **Default**. Confirm `3 + 5` is right and `3 + 50` warns OOD.
2. Switch to **Spicy**, retrain, compare curves and smartness.
3. Drop learning rate to `0.001`, raise epochs, watch a slower climb.
4. Ask: if we wanted `50` to work, what would we change — encoding, vocabulary, or task?

The honest answer to that last question is the whole blog post in one line: **a model only learns the world you can represent.**
