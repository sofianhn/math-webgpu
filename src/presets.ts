/**
 * Empirically chosen Simple / Mid / Ninja defaults.
 *
 * Hardened for mul/carry (user fail: 47×19 → 813 vs 893):
 * - NINJA_MIX overweight × + carry-heavy +
 * - two-layer trunk + tens-digit loss weight 1.35
 * - sweep-mul.ts: config B got 47×19 ✓ @ p=1.0, hardMul ~92%, hold ~93%
 *
 * SIMPLE stays faster; Mid is the compromise; Ninja is the page default.
 */
export type ModeId = "simple" | "mid" | "ninja";

export type ModePreset = {
  id: ModeId;
  label: string;
  blurb: string;
  hidden: number;
  epochs: number;
  lr: number;
  examples: number;
  batch: number;
  /** Which data mix the mode uses when training fresh. */
  mix: "simple" | "ninja";
  hardBias: number;
};

export const MODE_PRESETS: Record<ModeId, ModePreset> = {
  simple: {
    id: "simple",
    label: "Simple",
    blurb: "Faster train · lighter mul mix · solid everyday arithmetic",
    hidden: 128,
    epochs: 60,
    lr: 0.01,
    examples: 4096,
    batch: 64,
    mix: "simple",
    hardBias: 0.25,
  },
  mid: {
    id: "mid",
    label: "Mid",
    blurb: "Middle ground · H192 · more mul bias than Simple, shorter than Ninja",
    hidden: 192,
    epochs: 100,
    lr: 0.006,
    examples: 8192,
    batch: 64,
    mix: "ninja",
    hardBias: 0.4,
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
    mix: "ninja",
    hardBias: 0.5,
  },
};

/** Default mode when the page loads — Ninja is the impressive story. */
export const DEFAULT_MODE: ModeId = "ninja";

/** Saved checkpoint ids match mode ids for the three pretrained artifacts. */
export const CHECKPOINT_IDS: ModeId[] = ["simple", "mid", "ninja"];
