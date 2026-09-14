from dataclasses import fields

from secrag.config import VARIANTS, PipelineConfig


def test_each_variant_changes_one_thing_from_the_previous():
    variants = list(VARIANTS.values())
    for prev, cur in zip(variants, variants[1:], strict=False):
        changed = [
            f.name
            for f in fields(PipelineConfig)
            if f.name != "name" and getattr(prev, f.name) != getattr(cur, f.name)
        ]
        assert len(changed) == 1, (cur.name, changed)
