#!/usr/bin/env python3
"""Dump Python AutoExtractor answers for the Fastino demo texts."""
from __future__ import annotations

import json
from pathlib import Path

from gliner2 import AutoExtractor

OUT = Path("/Users/nemb/projects/pastel-org/pastel-evals/results/gliner25-golden-python.json")


def main():
    model = AutoExtractor.from_pretrained("fastino/gliner2.5-small-v1", map_location="cpu")
    cases = {}

    cases["ner-card"] = model.extract_entities(
        "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday.",
        ["company", "person", "product", "location"],
        include_confidence=True,
        include_spans=True,
    )

    cases["cls-sentiment"] = model.classify_text(
        "This laptop has amazing performance but terrible battery life!",
        {"sentiment": ["positive", "negative", "neutral"]},
    )

    cases["cls-constrained"] = model.classify_text(
        "Cancel this order immediately. The device is defective and I want a refund today.",
        {
            "sentiment": ["positive", "negative", "neutral"],
            "urgency": ["urgent", "not_urgent"],
        },
    )

    cases["kg-blog"] = model.extract_relations(
        "Tim Cook leads Apple in Cupertino. Sundar Pichai runs Google in Mountain View.",
        {
            "leads": {"head": ["person"], "tail": ["organization"]},
            "located_in": {"head": ["organization"], "tail": ["location"]},
        },
        include_confidence=True,
        include_spans=True,
    )

    cases["json-macbook"] = model.extract_json(
        "The new MacBook Pro costs $1999 and has a stunning Liquid Retina display.",
        {"product": ["name::str", "price", "features"]},
        include_confidence=True,
    )

    cases["json-rx"] = model.extract_json(
        "Patient: Sarah Johnson, 34, presented with acute chest pain and shortness of breath. Prescribed: Lisinopril 10mg daily, Metoprolol 25mg twice daily. Follow-up scheduled for next Tuesday.",
        {"prescription": ["medication", "dosage", "frequency", "name::str", "age::str"]},
        include_confidence=True,
        include_spans=True,
    )

    from gliner2.inference.schema import AttributeGroup
    schema = model.create_schema()
    schema.entities(["product"]).entity_attributes({
        "sentiment": AttributeGroup(labels=["positive", "negative", "neutral"], applies_to=["product"]),
    })
    cases["attr-product"] = model.extract(
        "The iPhone camera is stunning but the battery life is disappointing, while the speakers are fine.",
        schema,
        include_confidence=True,
        include_spans=True,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, indent=2, default=str))
    print(f"wrote {OUT}")
    for k, v in cases.items():
        print(f"\n=== {k} ===")
        print(json.dumps(v, indent=2, default=str)[:1200])


if __name__ == "__main__":
    main()
