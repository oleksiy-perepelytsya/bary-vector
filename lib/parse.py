"""Pure kaikki-record → ParsedWord/ParsedSense extraction.

Kept I/O-free so unit tests can drive it with fixture dicts. Stage 01
streams the JSONL through :func:`parse_entry` and writes the results.
"""

from __future__ import annotations

from typing import Any

from lib.bary_vec import count_syllables
from lib.disambiguate import parse_dis1
from lib.schema import ParsedSense, ParsedSenseRelation, ParsedWord

# Kaikki relation field names we care about (word-level and sense-level).
RELATION_KINDS: tuple[str, ...] = (
    "antonyms",
    "meronyms",
    "holonyms",
    "hypernyms",
    "hyponyms",
    "derived",
    "related",
    "coordinate_terms",
    "synonyms",
)


def _extract_ipa(sounds: list[dict[str, Any]]) -> str | None:
    """First IPA entry without dialect tags; fall back to first IPA at all."""
    first: str | None = None
    for s in sounds:
        ipa = s.get("ipa")
        if not ipa:
            continue
        if first is None:
            first = ipa
        if not s.get("tags"):
            return ipa
    return first


def _extract_relations(obj: dict[str, Any]) -> list[ParsedSenseRelation]:
    out: list[ParsedSenseRelation] = []
    for kind in RELATION_KINDS:
        for item in obj.get(kind) or []:
            w = item.get("word")
            if not w:
                continue
            out.append(
                ParsedSenseRelation(kind=kind, word=w, dis1=parse_dis1(item.get("_dis1")))
            )
    return out


def _sense_embed_text(gloss: str, examples: list[str]) -> str:
    return " ".join([gloss, *examples[:2]]).strip()


def _make_sense_id(word: str, pos: str, idx: int, raw: dict[str, Any]) -> str:
    sid = raw.get("id")
    if sid:
        return str(sid)
    return f"{word}-{pos}-{idx}"


def parse_entry(obj: dict[str, Any]) -> tuple[ParsedWord, list[ParsedSense]] | None:
    """Convert one kaikki JSONL record to a ParsedWord + its ParsedSense list.

    Returns ``None`` for non-English entries or entries with no senses.
    """
    word = obj.get("word")
    pos = obj.get("pos")
    if not word or not pos:
        return None
    if (obj.get("lang_code") or "en") != "en":
        return None

    raw_senses = obj.get("senses") or []
    senses: list[ParsedSense] = []
    for idx, s in enumerate(raw_senses):
        glosses = s.get("glosses") or []
        gloss = glosses[0] if glosses else ""
        if not gloss:
            continue
        examples = [
            e.get("text", "") for e in (s.get("examples") or []) if e.get("text")
        ]
        senses.append(
            ParsedSense(
                word=word,
                pos=pos,
                sense_id=_make_sense_id(word, pos, idx, s),
                sense_idx=idx,
                gloss=gloss,
                examples=examples,
                tags=list(s.get("tags") or []),
                topics=list(s.get("topics") or []),
                wikidata=list(s.get("wikidata") or []),
                relations=_extract_relations(s),
                embed_text=_sense_embed_text(gloss, examples),
            )
        )
    if not senses:
        return None

    forms = [f.get("form", "") for f in (obj.get("forms") or []) if f.get("form")]
    pw = ParsedWord(
        word=word,
        pos=pos,
        lang_code=obj.get("lang_code") or "en",
        ipa=_extract_ipa(obj.get("sounds") or []),
        forms=forms,
        etymology=obj.get("etymology_text") or "",
        char_len=len(word),
        syllable_ct=count_syllables(word),
        sense_ids=[s.sense_id for s in senses],
        relations=_extract_relations(obj),
    )
    return pw, senses
