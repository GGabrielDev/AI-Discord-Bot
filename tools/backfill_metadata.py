import argparse
import asyncio
import sys
import time
from urllib.parse import urlparse

import chromadb

from config.settings import CHROMA_DB_PATH
from storage.vectordb import VectorDB


def _source_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _path_depth(url: str) -> int:
    try:
        path = urlparse(url).path.strip("/")
        return len([segment for segment in path.split("/") if segment])
    except Exception:
        return 0


def _quality_score(total_chunks: int, raw_chunks: int, summary_chunks: int) -> float:
    raw_score = 1.0 if raw_chunks > 0 else 0.0
    summary_score = 1.0 if summary_chunks > 0 else 0.0
    coverage_score = min(total_chunks, 8) / 8
    return round(min(1.0, 0.2 + (0.25 * raw_score) + (0.25 * summary_score) + (0.3 * coverage_score)), 3)


def _build_metadata_update(meta: dict, document: str, stats: dict, now_ts: int) -> dict:
    source = meta["source"]
    latest_ts = stats["latest_ts"]
    age_days = max(0, int((now_ts - latest_ts) / 86400)) if latest_ts else 0

    merged = dict(meta)
    merged.update({
        "source_domain": _source_domain(source),
        "source_path_depth": _path_depth(source),
        "chunk_word_count": len((document or "").split()),
        "chunk_char_count": len(document or ""),
        "source_total_chunks": stats["total"],
        "source_raw_chunks": stats["raw"],
        "source_summary_chunks": stats["summary"],
        "source_has_raw": 1 if stats["raw"] > 0 else 0,
        "source_has_summary": 1 if stats["summary"] > 0 else 0,
        "source_age_days": age_days,
        "source_quality_score": _quality_score(stats["total"], stats["raw"], stats["summary"]),
        "metadata_backfilled_at": str(now_ts)
    })
    return merged


def _metadata_changed(old: dict, new: dict) -> bool:
    comparable_new = dict(new)
    comparable_old = dict(old)
    comparable_new.pop("metadata_backfilled_at", None)
    comparable_old.pop("metadata_backfilled_at", None)
    return comparable_old != comparable_new


async def backfill_collection(collection_name: str, topic_name: str, dry_run: bool = True, batch_size: int = 100, force: bool = False):
    print(f"\n--- Backfilling Collection: {collection_name} ---")
    db = VectorDB(collection_name=topic_name)
    all_data = await db.get_all_chunks()
    documents = all_data.get("documents") or []
    metadatas = all_data.get("metadatas") or []
    ids = all_data.get("ids") or []

    if not ids:
        print("Collection is empty.")
        return

    source_stats = {}
    for meta in metadatas:
        if not meta:
            continue
        source = meta.get("source")
        if not source:
            continue
        stats = source_stats.setdefault(source, {"total": 0, "raw": 0, "summary": 0, "latest_ts": 0})
        stats["total"] += 1
        if meta.get("chunk_type") == "raw":
            stats["raw"] += 1
        if meta.get("chunk_type") == "summary":
            stats["summary"] += 1
        stats["latest_ts"] = max(stats["latest_ts"], int(meta.get("timestamp", 0) or 0))

    now_ts = int(time.time())
    update_ids = []
    update_metadatas = []
    skipped_unchanged = 0
    already_backfilled = 0
    changed_sources = set()
    for chunk_id, document, meta in zip(ids, documents, metadatas):
        if not meta:
            continue
        source = meta.get("source")
        if not source:
            continue
        stats = source_stats[source]
        merged = _build_metadata_update(meta, document or "", stats, now_ts)
        changed = _metadata_changed(meta, merged)
        if meta.get("metadata_backfilled_at"):
            already_backfilled += 1
        if not force and not changed:
            skipped_unchanged += 1
            continue

        update_ids.append(chunk_id)
        update_metadatas.append(merged)
        changed_sources.add(source)

    print(f"Total chunks: {len(ids)}")
    print(f"Unique sources: {len(source_stats)}")
    print(f"Chunks to update: {len(update_ids)}")
    print(f"Unchanged chunks skipped: {skipped_unchanged}")
    print(f"Previously backfilled chunks seen: {already_backfilled}")
    print(f"Sources impacted: {len(changed_sources)}")

    if dry_run:
        print("DRY RUN: No metadata updated. Use --execute to apply changes.")
        if not force:
            print("Tip: use --force only if you intentionally want to refresh already-derived metadata.")
        print("Safety note: back up `chroma_data/` before first production execute.")
        return

    for i in range(0, len(update_ids), batch_size):
        await db.update_chunk_metadata(
            update_ids[i:i + batch_size],
            update_metadatas[i:i + batch_size]
        )

    print("Metadata backfill complete.")
    if not update_ids:
        print("Nothing changed.")


async def main_async():
    parser = argparse.ArgumentParser(description="Backfill derived retrieval metadata for ChromaDB chunks.")
    parser.add_argument("--topic", type=str, help="Specific topic (collection) to annotate. If omitted, annotates all collections.")
    parser.add_argument("--execute", action="store_true", help="Apply metadata updates (default is dry-run).")
    parser.add_argument("--batch-size", type=int, default=100, help="Chroma update batch size (default: 100).")
    parser.add_argument("--force", action="store_true", help="Rewrite metadata even when derived fields already match.")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collections = client.list_collections()

    if args.topic:
        target = [c for c in collections if c.name == args.topic or c.name == args.topic.replace(" ", "_")]
        if not target:
            print(f"Error: Collection '{args.topic}' not found.")
            sys.exit(1)
        await backfill_collection(target[0].name, args.topic, not args.execute, args.batch_size, args.force)
        return

    for col in collections:
        await backfill_collection(col.name, col.name, not args.execute, args.batch_size, args.force)


if __name__ == "__main__":
    asyncio.run(main_async())
