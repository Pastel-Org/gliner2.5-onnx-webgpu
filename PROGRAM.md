# GLiNER2.5 ONNX + JS + Weightlift program

Four layers. No new training. Neural scoring in ONNX, combinatorial
decode in JS. Encoder runs once per extract.

## 1. ONNX graph (fused, extra I/O)

Public checkpoints already contain `classifier`, `relation_scorer`, and
`RecordHead`. v2 only traces encoder + boundary + pair scorer.

| Version | In the graph | Host |
|---|---|---|
| v2 (shipped) | encoder, boundary, pair scorer | entity decode |
| v3 | + `classifier` + emit `text_states` / `query_states` | classify + attribute rescore |
| v4 | + vectorized relation pair generator + scorer | JointIE beam in JS |
| v5 | + record assignment logits | instance formation in JS |

v3 inputs add `cls_marker_indices` / `cls_marker_mask` (Python already
builds these). Dummy K=1 masked when unused. Do not ship a second encoder.

## 2. JS library (`src/api.mjs`)

Python `AutoExtractor` shape: `extract_entities`, `extract_json`,
`classify_text`, `classify_text_long`, `extract_relations`, `extract_*_long`.
`classify_text` is one shot (WebGPU base/multi die ~3500 words).
`classify_text_long` is 384/64, same as Python.

## 3. Weightlift adapter (`src/weightlift.mjs`)

`glinerModel({ size })` returns a weightlift `ModelDefinition`:
`load({ progress })` → `Gliner25` handle. Lifecycle only. We do **not**
control [weightlift.dev](https://weightlift.dev) (wassgha/weightlift).
The playground can import this helper the same way it imports
`transformersModel`. A PR to their `demo/` is optional and separate.

## 4. Demo page

`ModelManager` loads small/base/multi. Fastino README/tutorial/blog
examples as blank templates. Extra Pastel templates (PII, long clause,
chunked contract). Blocked heads stay labelled until the matching graph
version lands.
