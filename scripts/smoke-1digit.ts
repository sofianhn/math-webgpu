import { CpuBackend } from "../src/webgpu";
import { TinyMathMlp } from "../src/model";
import {
  ANSWER_DIGITS,
  ArithmeticDataset,
  MathDataPipeline,
  BALANCED_MIX,
  computeAnswer,
  encodeFeatures,
  toDigits,
  type Op,
} from "../src/data";
import { MathTrainer } from "../src/trainer";
import {
  batchFromExprs,
  buildExpressionHoldout,
  evaluateModel,
  isExpressionHeldOut,
} from "../src/evals";

async function main() {
  const gpu = new CpuBackend();
  const model = new TinyMathMlp(gpu, 192);
  const filter = (a: bigint, b: bigint, op: Op) => !isExpressionHeldOut(a, op, b);
  const ds = new ArithmeticDataset({ filter, mix: BALANCED_MIX, hardBias: 0.2, maxDigits: 1 });
  const pipe = new MathDataPipeline("pretrain", ds);
  const hold = buildExpressionHoldout(120, 10, 1);
  const trainer = new MathTrainer();
  console.log(`1-digit only · hold n=${hold.length}`);
  await trainer.train(model, pipe, {
    stage: "pretrain",
    epochs: 50,
    batchSize: 64,
    learningRate: 0.01,
    stepsPerEpoch: 64,
    heldoutExprs: hold,
    curriculumStartDigits: 1,
    curriculumEndDigits: 1,
    onCurriculumDigits: (d) => ds.setMaxDigits(d),
    onMetrics: (m) => {
      if (m.epoch === 1 || m.epoch % 10 === 0 || m.epoch === 50) {
        const h = m.heldout
          ? ` hold=${(m.heldout.accuracy * 100).toFixed(1)}% dig=${(m.heldout.digitAccuracy * 100).toFixed(1)}%`
          : "";
        console.log(`ep ${m.epoch} exact=${(m.accuracy * 100).toFixed(1)}% dig=${(m.digitAccuracy * 100).toFixed(1)}%${h}`);
      }
    },
  });
  const final = await evaluateModel(model, batchFromExprs(hold));
  console.log(`FINAL hold exact=${(final.accuracy * 100).toFixed(1)}% digit=${(final.digitAccuracy * 100).toFixed(1)}%`);
  for (const [a, op, b] of [[3n, "+", 5n], [7n, "*", 8n], [9n, "-", 4n], [8n, "/", 2n]] as const) {
    const t = computeAnswer(a, op, b);
    const { answer, digits, digitProbs } = await model.predict(encodeFeatures(a, op, b));
    const truthDigs = [...toDigits(t, ANSWER_DIGITS)].slice(0, 4).join("");
    const predDigs = digits.slice(0, 4).join("");
    const conf = digits.slice(0, 4).map((d, p) => digitProbs[p * 10 + d]!.toFixed(2)).join(",");
    console.log(`${a}${op}${b}=${t} → ${answer} digs=${predDigs} want=${truthDigs} conf=${conf} ${answer === t ? "✓" : "✗"}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
