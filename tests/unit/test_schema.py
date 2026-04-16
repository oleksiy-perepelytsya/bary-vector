from __future__ import annotations

import orjson

from lib.schema import ParsedSense, ParsedSenseRelation, ParsedWord


def test_parsed_sense_roundtrip():
    s = ParsedSense(
        word="happy",
        pos="adj",
        sense_id="happy-en-adj-1",
        sense_idx=0,
        gloss="feeling pleasure",
        examples=["she was happy"],
        tags=[],
        relations=[ParsedSenseRelation(kind="synonyms", word="glad", sense_idx=0)],
        embed_text="feeling pleasure she was happy",
    )
    d = s.to_dict()
    back = orjson.loads(orjson.dumps(d))
    assert back["word"] == "happy"
    assert back["relations"][0]["kind"] == "synonyms"


def test_parsed_word_roundtrip():
    w = ParsedWord(word="happy", pos="adj", lang_code="en", ipa="/ˈhæpi/",
                   sense_ids=["happy-en-adj-1", "happy-en-adj-2"])
    assert w.to_dict()["sense_ids"] == ["happy-en-adj-1", "happy-en-adj-2"]
