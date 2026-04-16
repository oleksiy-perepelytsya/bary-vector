from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any

from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from lib.config import Settings

# Standard (non-vector) indexes — see CLAUDE.md.
STANDARD_INDEXES: list[list[tuple[str, int]]] = [
    [("doc_type", ASCENDING), ("level", ASCENDING)],
    [("cm1_id", ASCENDING)],
    [("cm2_id", ASCENDING)],
    [("node_type", ASCENDING)],
    [("edge_type", ASCENDING), ("level", ASCENDING)],
    [("parent_edge_id", ASCENDING)],
    [("properties.word", ASCENDING), ("properties.pos", ASCENDING)],
    [("properties.sense_id", ASCENDING)],
]


@lru_cache(maxsize=8)
def _cached_client(uri: str) -> MongoClient:
    return MongoClient(uri, serverSelectionTimeoutMS=5000)


def get_client(settings: Settings) -> MongoClient:
    return _cached_client(settings.mongo_uri)


def get_collection(settings: Settings) -> Collection:
    return get_client(settings)[settings.mongo_db][settings.mongo_collection]


def ping(settings: Settings) -> bool:
    try:
        get_client(settings).admin.command("ping")
        return True
    except PyMongoError:
        return False


def ensure_indexes(coll: Collection) -> list[str]:
    names: list[str] = []
    for keys in STANDARD_INDEXES:
        names.append(coll.create_index(keys))
    return names


def bulk_upsert(coll: Collection, docs: Iterable[Mapping[str, Any]], key: str = "_id") -> int:
    ops = [UpdateOne({key: d[key]}, {"$set": dict(d)}, upsert=True) for d in docs]
    if not ops:
        return 0
    res = coll.bulk_write(ops, ordered=False)
    return (res.upserted_count or 0) + (res.modified_count or 0)
