import { softmax, cropWordStates, resolveOverlapsFlat, iterWordChunks, normalizeText } from "../src/gliner-boundary.mjs";

const p = softmax([1, 2, 3]);
const sum = p.reduce((a, b) => a + b, 0);
if (Math.abs(sum - 1) > 1e-12) throw new Error(`softmax sum ${sum}`);
if (!(p[2] > p[1] && p[1] > p[0])) throw new Error(`softmax order ${p}`);

const data = new Float32Array(20 * 4);
for (let i = 0; i < data.length; i++) data[i] = i;
const cropped = cropWordStates({ dims: [1, 20, 4], data }, [{ wordStart: 12, wordEnd: 14 }], 8);
if (cropped.origin !== 9) throw new Error(`origin ${cropped.origin}`);
if (cropped.ts[0] !== 36) throw new Error(`crop start ${cropped.ts[0]}`);

const kept = resolveOverlapsFlat([
  { start: 1, end: 2, score: 0.7911 },
  { start: 1, end: 3, score: 0.9030 },
]);
if (kept.length !== 1 || kept[0].start !== 1 || kept[0].end !== 3) {
  throw new Error(`overlap kept ${JSON.stringify(kept)}`);
}

// ONNX pair_valid is all-true; padded slots like [16,5) @ 1.0 must not steal the span.
const polluted = resolveOverlapsFlat([
  { start: 1, end: 2, score: 0.7911 },
  { start: 1, end: 3, score: 0.9030 },
  { start: 16, end: 5, score: 1 },
  { start: 14, end: 5, score: 1 },
  { start: 7, end: 3, score: 1 },
].filter((x) => x.start < x.end));
if (polluted.length !== 1 || polluted[0].end !== 3) {
  throw new Error(`filtered overlap ${JSON.stringify(polluted)}`);
}

const padded = Array.from({ length: 800 }, (_, i) => `w${i}`).join(" ") + ".";
const { words, chunks } = iterWordChunks(normalizeText(padded), { chunkSize: 384, chunkOverlap: 64 });
if (words.length !== 801) throw new Error(`words ${words.length}`);
if (chunks.length !== 3) throw new Error(`chunks ${chunks.length}`);
if (chunks[0].wordStart !== 0 || chunks[0].wordEnd !== 384) throw new Error(`c0 ${JSON.stringify(chunks[0])}`);
if (chunks[1].wordStart !== 320 || chunks[1].wordEnd !== 704) throw new Error(`c1 ${JSON.stringify(chunks[1])}`);
if (chunks[2].wordStart !== 640 || chunks[2].wordEnd !== 801) throw new Error(`c2 ${JSON.stringify(chunks[2])}`);
console.log("ok");
