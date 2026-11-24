/**
 * Compact checkpoint format: weights + hyperparams + train/eval curves.
 * Weights are Float32 → base64 (binary-safe, ~1.33× raw size).
 */
import type { EvalSliceMetrics, TrainMetrics } from "./types";
import type { TinyMathMlp } from "./model";
import { INPUT_DIM, NUM_CLASSES } from "./data";

export const CHECKPOINT_VERSION = 1 as const;

export type CheckpointId = "simple" | "mid" | "ninja";

export type CheckpointHyperparams = {
  hidden: number;
  epochs: number;
  lr: number;
  examples: number;
  batch: number;
  /** Op mix label for UI. */
  mix: "simple" | "ninja";
  hardBias: number;
  heldout: boolean;
};

export type CheckpointMeta = {
  version: typeof CHECKPOINT_VERSION;
  id: CheckpointId;
  label: string;
  blurb: string;
  createdAt: string;
  arch: {
    name: string;
    hidden: number;
    inputDim: number;
    numClasses: number;
    paramCount: number;
  };
  hyperparams: CheckpointHyperparams;
  holdout: {
    pct: number;
    n: number;
    seed: number;
  };
  /** Per-epoch train + held-out metrics (frozen probe set). */
  curves: TrainMetrics[];
  /** Final frozen held-out eval after last epoch. */
  finalHoldout: EvalSliceMetrics;
  /** Optional probe pack accuracies (easy / hard / curriculum). */
  probes?: {
    easy: number;
    hard: number;
    curriculum: number;
  };
};

export type CheckpointFile = CheckpointMeta & {
  weights: {
    encoding: "f32-b64";
    /** Param name → base64 of little-endian Float32 bytes. */
    params: Record<string, string>;
  };
};

/** Manifest listing available checkpoints (no weight payloads). */
export type CheckpointManifest = {
  version: typeof CHECKPOINT_VERSION;
  models: Array<{
    id: CheckpointId;
    label: string;
    blurb: string;
    path: string;
    hidden: number;
    epochs: number;
    paramCount: number;
    finalHoldoutAcc: number;
  }>;
};

function float32ToB64(data: Float32Array): string {
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  let bin = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode(...bytes.subarray(i, Math.min(i + chunk, bytes.length)));
  }
  return btoa(bin);
}

function b64ToFloat32(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

/** Export live model weights into a serializable params map. */
export function exportWeights(model: TinyMathMlp): Record<string, string> {
  const params: Record<string, string> = {};
  for (const { name, tensor } of model.parameters()) {
    params[name] = float32ToB64(tensor.data);
  }
  return params;
}

/** Copy checkpoint weights into an existing model (same hidden size). */
export function loadWeights(model: TinyMathMlp, params: Record<string, string>): void {
  for (const { name, tensor } of model.parameters()) {
    const b64 = params[name];
    if (!b64) throw new Error(`checkpoint missing weight "${name}"`);
    const data = b64ToFloat32(b64);
    if (data.length !== tensor.data.length) {
      throw new Error(
        `weight "${name}" length mismatch: ckpt ${data.length} vs model ${tensor.data.length}`,
      );
    }
    tensor.data.set(data);
  }
}

export function buildCheckpointMeta(opts: {
  id: CheckpointId;
  label: string;
  blurb: string;
  hyperparams: CheckpointHyperparams;
  holdout: CheckpointMeta["holdout"];
  curves: TrainMetrics[];
  finalHoldout: EvalSliceMetrics;
  probes?: CheckpointMeta["probes"];
  paramCount: number;
}): CheckpointMeta {
  return {
    version: CHECKPOINT_VERSION,
    id: opts.id,
    label: opts.label,
    blurb: opts.blurb,
    createdAt: new Date().toISOString(),
    arch: {
      name: "digit-math-mlp",
      hidden: opts.hyperparams.hidden,
      inputDim: INPUT_DIM,
      numClasses: NUM_CLASSES,
      paramCount: opts.paramCount,
    },
    hyperparams: opts.hyperparams,
    holdout: opts.holdout,
    curves: opts.curves,
    finalHoldout: opts.finalHoldout,
    probes: opts.probes,
  };
}

export function packCheckpoint(meta: CheckpointMeta, model: TinyMathMlp): CheckpointFile {
  return {
    ...meta,
    weights: {
      encoding: "f32-b64",
      params: exportWeights(model),
    },
  };
}

/** Strip weights for UI summaries / smaller fetches of metadata-only views. */
export function stripWeights(ckpt: CheckpointFile): CheckpointMeta {
  const { weights: _w, ...meta } = ckpt;
  return meta;
}

export function checkpointUrl(id: CheckpointId, base = import.meta.env.BASE_URL): string {
  const root = base.endsWith("/") ? base : `${base}/`;
  return `${root}checkpoints/${id}.json`;
}

export function manifestUrl(base = import.meta.env.BASE_URL): string {
  const root = base.endsWith("/") ? base : `${base}/`;
  return `${root}checkpoints/manifest.json`;
}

export async function fetchCheckpoint(id: CheckpointId): Promise<CheckpointFile> {
  const res = await fetch(checkpointUrl(id));
  if (!res.ok) throw new Error(`Failed to load checkpoint ${id}: HTTP ${res.status}`);
  const ckpt = (await res.json()) as CheckpointFile;
  if (ckpt.version !== CHECKPOINT_VERSION) {
    throw new Error(`Unsupported checkpoint version ${ckpt.version}`);
  }
  return ckpt;
}

export async function fetchManifest(): Promise<CheckpointManifest | null> {
  try {
    const res = await fetch(manifestUrl());
    if (!res.ok) return null;
    return (await res.json()) as CheckpointManifest;
  } catch {
    return null;
  }
}
