/**
 * Empirically chosen Simple vs Ninja defaults.
 *
 * Hardened for mul/carry (user fail: 47×19 → 813 vs 893):
 * - NINJA_MIX overweight × + carry-heavy +
 * - two-layer trunk + tens-digit loss weight 1.35
 * - sweep-mul.ts: config B got 47×19 ✓ @ p=1.0, hardMul ~92%, hold ~93%
 *
 * SIMPLE stays faster; Ninja is the page default.
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
    blurb: "Faster train · lighter mul mix · solid everyday arithmetic",
    hidden: 128,
    epochs: 60,
    lr: 0.01,
    examples: 4096,
    batch: 64,
  },
  ninja: {
    id: "ninja",
    label: "Ninja",
    blurb:
      "Default · mul/carry-weighted data, deeper trunk, H288 — tuned so hard products like 47×19 land",
    hidden: 288,
    epochs: 160,
    lr: 0.0045,
    examples: 12288,
    batch: 64,
  },
};

/** Default mode when the page loads — Ninja is the impressive story. */
export const DEFAULT_MODE: "simple" | "ninja" = "ninja";
