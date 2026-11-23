/**
 * Empirically chosen Simple vs Ninja defaults (CPU sweeps, 3 seeds).
 *
 * Metrics are fair held-out (~10% frozen expressions) + fixed easy/hard probes.
 * See scripts/sweep-presets.ts, sweep-refine.ts, sweep-confirm.ts.
 *
 * SIMPLE+  H=112 ep=55 lr=0.01 ex=3072  → ~84% train / ~83% hold
 * NINJA+   H=256 ep=150 lr=0.005 ex=10240 → ~96% train / ~90% hold / ~83% hard probes
 */
export type ModePreset = {
  id: "simple" | "ninja";
  label: string;
  blurb: string;
  hidden: number;
  epochs: number;
  lr: number;
  examples: number;
  batch: number;
};

export const MODE_PRESETS: Record<"simple" | "ninja", ModePreset> = {
  simple: {
    id: "simple",
    label: "Simple",
    blurb: "Faster train · solid on everyday sums (~83% held-out in sweeps)",
    hidden: 112,
    epochs: 55,
    lr: 0.01,
    examples: 3072,
    batch: 64,
  },
  ninja: {
    id: "ninja",
    label: "Ninja",
    blurb: "Default · heavy data + capacity for hard mul/carry (~90% held-out, ~83% hard probes)",
    hidden: 256,
    epochs: 150,
    lr: 0.005,
    examples: 10240,
    batch: 64,
  },
};

/** Default mode when the page loads — Ninja is the impressive story. */
export const DEFAULT_MODE: "simple" | "ninja" = "ninja";
