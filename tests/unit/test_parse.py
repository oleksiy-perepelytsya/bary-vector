from __future__ import annotations

import orjson

from lib.parse import RELATION_KINDS, _extract_ipa, parse_entry

FIXTURE = "tests/fixtures/kaikki-sample.jsonl"


def _records():
    with open(FIXTURE, "rb") as f:
        return [orjson.loads(line) for line in f]


def test_parse_happy_word():
    rec = next(r for r in _records() if r["word"] == "happy")
    pw, senses = parse_entry(rec)
    assert pw.word == "happy" and pw.pos == "adj"
    assert pw.ipa == "/ˈhæpi/"
    assert len(senses) == 2
    assert all(s.embed_text for s in senses)
    assert pw.sense_ids == [s.sense_id for s in senses]
    kinds = {r.kind for r in pw.relations}
    assert "synonyms" in kinds and "antonyms" in kinds


def test_parse_filters_non_english():
    assert parse_entry({"word": "x", "pos": "n", "lang_code": "fr",
                        "senses": [{"glosses": ["y"]}]}) is None


def test_parse_skips_empty_senses():
    assert parse_entry({"word": "x", "pos": "n", "lang_code": "en", "senses": []}) is None


def test_parse_generates_sense_id_when_missing():
    rec = {"word": "foo", "pos": "noun", "lang_code": "en",
           "senses": [{"glosses": ["a thing"]}]}
    _, senses = parse_entry(rec)
    assert senses[0].sense_id == "foo-noun-0"


def test_embed_text_uses_gloss_plus_first_two_examples():
    rec = {"word": "foo", "pos": "n", "lang_code": "en",
           "senses": [{"glosses": ["G"],
                       "examples": [{"text": "e1"}, {"text": "e2"}, {"text": "e3"}]}]}
    _, senses = parse_entry(rec)
    assert senses[0].embed_text == "G e1 e2"


def test_extract_ipa_prefers_untagged():
    sounds = [{"ipa": "/a/", "tags": ["US"]}, {"ipa": "/b/"}]
    assert _extract_ipa(sounds) == "/b/"
    assert _extract_ipa([{"ipa": "/a/", "tags": ["UK"]}]) == "/a/"
    assert _extract_ipa([{"audio": "x.ogg"}]) is None


def test_relation_kinds_cover_fermion_fields():
    from lib.match import FERMION_TIERS
    needed = {f for t in FERMION_TIERS for f in t.kaikki_fields}
    assert needed <= set(RELATION_KINDS)
