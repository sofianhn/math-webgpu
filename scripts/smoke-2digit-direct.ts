import { CpuBackend } from "../src/webgpu";
import { TinyMathMlp } from "../src/model";
import {
  ArithmeticDataset,
  MathDataPipeline,
  NINJA_MIX,
  computeAnswer,
  encodeFeatures,
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
  const model = new TinyMathMlp(gpu, 288);
  const filter = (a: bigint, b: bigint, op: Op) => !isExpressionHeldOut(a, op, b);
  const ds = new ArithmeticDataset({ filter, mix: NINJA_MIX, hardBias: 0.5, maxDigits: 2 });
  const pipe = new MathDataPipeline("pretrain", ds);
  const hold = buildExpressionHoldout(200, 10, 2);
  const trainer = new MathTrainer();
  const epochs = 80;
  const batch = 64;
  const steps = 96;
  console.log(`2-digit DIRECT · H=288 · ep=${epochs} · steps=${steps}`);
  await trainer.train(model, pipe, {
    stage: "pretrain",
    epochs,
    batchSize: batch,
    learningRate: 0.005,
    stepsPerEpoch: steps,
    heldoutExprs: hold,
    curriculumStartDigits: 2,
    curriculumEndDigits: 2,
    onCurriculumDigits: (d) => ds.setMaxDigits(d),
    onMetrics: (m) => {
      if (m.epoch === 1 || m.epoch % 10 === 0 || m.epoch === epochs) {
        const h = m.heldout
          ? ` hold=${(m.heldout.accuracy * 100).toFixed(1)}% dig=${(m.heldout.digitAccuracy * 100).toFixed(1)}%`
          : "";
        console.log(
          `ep ${m.epoch} exact=${(m.accuracy * 100).toFixed(1)}% dig=${(m.digitAccuracy * 100).toFixed(1)}%${h}`,
        );
      }
    },
  });
  const final = await evaluateModel(model, batchFromExprs(hold));
  console.log(`FINAL hold exact=${(final.accuracy * 100).toFixed(1)}% digit=${(final.digitAccuracy * 100).toFixed(1)}%`);
  for (const [a, op, b] of [[3n,"+",5n],[12n,"+",34n],[47n,"+",38n],[17n,"*",19n],[47n,"*",19n],[25n,"*",25n],[99n,"-",17n],[84n,"/",4n]] as const) {
    const t = computeAnswer(a, op, b);
    const { answer } = await model.predict(encodeFeatures(a, op, b));
    console.log(`${a}${op}${b}=${t} → ${answer} ${answer === t ? "✓" : "✗"}`);
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
