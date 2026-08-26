import { softmax, cropWordStates } from "../src/gliner-boundary.mjs";

const p = softmax([1, 2, 3]);
const sum = p.reduce((a, b) => a + b, 0);
if (Math.abs(sum - 1) > 1e-12) throw new Error(`softmax sum ${sum}`);
if (!(p[2] > p[1] && p[1] > p[0])) throw new Error(`softmax order ${p}`);

const data = new Float32Array(20 * 4);
for (let i = 0; i < data.length; i++) data[i] = i;
const cropped = cropWordStates({ dims: [1, 20, 4], data }, [{ wordStart: 12, wordEnd: 14 }], 8);
if (cropped.origin !== 9) throw new Error(`origin ${cropped.origin}`);
if (cropped.ts[0] !== 36) throw new Error(`crop start ${cropped.ts[0]}`);
console.log("ok");
