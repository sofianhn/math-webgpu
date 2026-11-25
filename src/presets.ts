/**
 * Simple / Mid / Ninja defaults for the nine-digit place-value calculator.
 *
 * Curriculum grows operand length 1 → endDigits; Ninja reaches full 9 digits
 * with mul/carry-weighted data and a wider trunk.
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
  /** Curriculum start / end operand digit lengths. */
  curriculumStart: number;
  curriculumEnd: number;
};

export const MODE_PRESETS: Record<ModeId, ModePreset> = {
  simple: {
    id: "simple",
    label: "Simple",
    blurb: "Faster train · staircase curriculum to 4 digits · lighter mul mix",
    hidden: 256,
    // 4 stages × ~25 epochs; short-length rehearsal prevents forgetting.
    epochs: 100,
    lr: 0.007,
    examples: 8192,
    batch: 64,
    mix: "simple",
    hardBias: 0.3,
    curriculumStart: 1,
    curriculumEnd: 4,
  },
  mid: {
    id: "mid",
    label: "Mid",
    blurb: "Middle ground · H384 · staircase curriculum to 7 digits",
    hidden: 384,
    epochs: 175,
    lr: 0.0045,
    examples: 12288,
    batch: 64,
    mix: "ninja",
    hardBias: 0.45,
    curriculumStart: 1,
    curriculumEnd: 7,
  },
  ninja: {
    id: "ninja",
    label: "Ninja",
    blurb:
      "Default · nine-digit staircase curriculum, mul/carry-weighted data, H512 shared digit head",
    hidden: 512,
    epochs: 270,
    lr: 0.003,
    examples: 16384,
    batch: 64,
    mix: "ninja",
    hardBias: 0.55,
    curriculumStart: 1,
    curriculumEnd: 9,
  },
};

/** Default mode when the page loads — Ninja is the impressive story. */
export const DEFAULT_MODE: ModeId = "ninja";

/** Saved checkpoint ids match mode ids for the three pretrained artifacts. */
export const CHECKPOINT_IDS: ModeId[] = ["simple", "mid", "ninja"];
