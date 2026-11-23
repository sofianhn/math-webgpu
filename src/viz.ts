import type { TrainMetrics } from "./types";

const PARAM_BLURB: Record<string, string> = {
  W1: "weights from input features → hidden units",
  b1: "bias added to each hidden unit",
  W2: "weights from hidden units → answer classes",
  b2: "bias added to each answer class",
};

/** SVG dual-axis loss + accuracy curve (educational training internals). */
export function renderCurves(
  svg: SVGSVGElement,
  history: TrainMetrics[],
  opts?: { title?: string },
): void {
  const W = 560;
  const H = 200;
  const pad = { t: 28, r: 48, b: 32, l: 44 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", opts?.title ?? "Training loss and accuracy");

  if (history.length === 0) {
    svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="#5c5a52" font-size="12">Train to see loss &amp; accuracy evolve</text>`;
    return;
  }

  const losses = history.map((m) => m.loss);
  const accs = history.map((m) => m.accuracy);
  const maxLoss = Math.max(...losses, 1e-6);
  const n = history.length;
  const xAt = (i: number) => pad.l + (n === 1 ? iw / 2 : (i / (n - 1)) * iw);
  const yLoss = (v: number) => pad.t + ih - (v / maxLoss) * ih;
  const yAcc = (v: number) => pad.t + ih - v * ih;

  const lossPath = losses.map((v, i) => `${i === 0 ? "M" : "L"}${xAt(i).toFixed(1)},${yLoss(v).toFixed(1)}`).join(" ");
  const accPath = accs.map((v, i) => `${i === 0 ? "M" : "L"}${xAt(i).toFixed(1)},${yAcc(v).toFixed(1)}`).join(" ");

  const last = history[history.length - 1]!;
  svg.innerHTML = `
    <text x="${pad.l}" y="16" fill="#1a1a18" font-size="12" font-weight="600">${opts?.title ?? "Training curves"}</text>
    <text x="${W - pad.r}" y="16" text-anchor="end" fill="#5c5a52" font-size="11">epoch ${last.epoch} · loss ${last.loss.toFixed(3)} · acc ${(last.accuracy * 100).toFixed(0)}%</text>
    <line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${pad.t + ih}" stroke="#d9d2c3"/>
    <line x1="${pad.l}" y1="${pad.t + ih}" x2="${pad.l + iw}" y2="${pad.t + ih}" stroke="#d9d2c3"/>
    <path d="${lossPath}" fill="none" stroke="#0b6e4f" stroke-width="2"/>
    <path d="${accPath}" fill="none" stroke="#b45a1a" stroke-width="2" stroke-dasharray="4 3"/>
    <text x="${pad.l}" y="${H - 8}" fill="#0b6e4f" font-size="10">━ loss</text>
    <text x="${pad.l + 52}" y="${H - 8}" fill="#b45a1a" font-size="10">╌ accuracy</text>
    <text x="${pad.l - 6}" y="${pad.t + 4}" text-anchor="end" fill="#5c5a52" font-size="9">${maxLoss.toFixed(2)}</text>
    <text x="${pad.l - 6}" y="${pad.t + ih}" text-anchor="end" fill="#5c5a52" font-size="9">0</text>
    <text x="${pad.l + iw + 6}" y="${pad.t + 4}" fill="#b45a1a" font-size="9">100%</text>
    <text x="${pad.l + iw + 6}" y="${pad.t + ih}" fill="#b45a1a" font-size="9">0%</text>
  `;
}

/**
 * Parameter tables: count of numbers stored + how much their “size” (L2) moved.
 * Educational: these are not hyperparameters (lr/epochs) — they are the model’s memory.
 */
export function renderNormBars(
  el: HTMLElement,
  before: { name: string; l2: number; count: number }[],
  after: { name: string; l2: number; count: number }[],
): void {
  const max = Math.max(...before.map((b) => b.l2), ...after.map((a) => a.l2), 1e-6);
  const total = after.reduce((s, a) => s + a.count, 0);
  el.innerHTML =
    `<p class="norm-intro">The network stores <strong>${total.toLocaleString()}</strong> learnable numbers in four tables. Bars = table “size” (L2 norm) before → after training; Δ shows how much that table moved.</p>` +
    before
      .map((b, i) => {
        const a = after[i]!;
        const bw = (b.l2 / max) * 100;
        const aw = (a.l2 / max) * 100;
        const delta = a.l2 - b.l2;
        const blurb = PARAM_BLURB[b.name] ?? "";
        return `<div class="norm-row">
        <div class="norm-meta">
          <span class="norm-name">${b.name}</span>
          <span class="norm-count">${a.count.toLocaleString()} numbers</span>
          <span class="norm-blurb">${blurb}</span>
        </div>
        <div class="norm-tracks">
          <div class="norm-bar before" style="width:${bw.toFixed(1)}%" title="before L2 ${b.l2.toFixed(3)}"></div>
          <div class="norm-bar after" style="width:${aw.toFixed(1)}%" title="after L2 ${a.l2.toFixed(3)}"></div>
        </div>
        <span class="norm-delta">${delta >= 0 ? "+" : ""}${delta.toFixed(2)}</span>
      </div>`;
      })
      .join("");
}

/** Live strip of training expressions the model just saw. */
export function renderDataFeed(el: HTMLElement, texts: string[], headline: string): void {
  el.innerHTML =
    `<p class="data-headline">${headline}</p>` +
    `<ul class="data-list">${texts.map((t) => `<li><code>${escapeHtml(t)}</code></li>`).join("")}</ul>`;
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** Top-k class probabilities as a compact list. */
export function renderTopK(el: HTMLElement, probs: Float32Array, k = 5): void {
  const idx = Array.from(probs.keys()).sort((i, j) => probs[j]! - probs[i]!);
  el.innerHTML = idx
    .slice(0, k)
    .map((i) => {
      const p = probs[i]!;
      return `<div class="topk-row"><span class="topk-cls">${i}</span>
        <div class="topk-track"><div class="topk-fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
        <span class="topk-p">${(p * 100).toFixed(1)}%</span></div>`;
    })
    .join("");
}
