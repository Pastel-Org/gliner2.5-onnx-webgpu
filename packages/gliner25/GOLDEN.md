# Python AutoExtractor goldens (small, 2026-08-26)

Source: `golden_python.py` → `fastino/gliner2.5-small-v1` on CPU.
JS should match these answers, not a prettier host decode.

## ner-card

Apple [0,5) 0.994 · Tim Cook [10,18) 0.999 · iPhone 15 [29,38) 0.999 · Cupertino [42,51) 0.999

## cls-sentiment / cls-constrained

`sentiment: negative`. Multi-task: `negative` + `urgent`.
Python `classify_text` has no `implies`/`excludes` argument. Our JS beam is extra.

## kg-blog (`extract_relations`)

One edge: Tim Cook —leads→ Apple (head/tail confidence **0.627**). `located_in` empty.
Pichai/Google is not returned.

## json-macbook

One product: name MacBook Pro 0.999, price $1999 0.964, features Liquid Retina display 0.910

## json-rx (RecordHead, first field = medication)

Three instances. Dosage and frequency empty on all. Names are long/low-score except one “Sarah Johnson” 0.717. Age once “34” 0.994 and two junk spans. Overlapping meds: Lisinopril, Metoprolol, Metoprolol 25mg.

## attr-product

One product: “iPhone camera” 0.903 with sentiment **negative 0.592** (softmax).
NER-only on the same text keeps “iPhone” 0.933. The joint `[E]` pack
(product + negative/neutral/positive, sorted) is what changes the span.
ONNX `pair_valid` is all-true; decode drops `start >= end` slots before overlap
or they steal the longer mention.
