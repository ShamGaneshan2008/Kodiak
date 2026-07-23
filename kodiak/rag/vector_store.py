"""
kodiak/rag/vector_store.py

ChromaDB-backed vector store for chunk embeddings.
Supports collection-per-repository isolation, metadata filtering,
and async wrappers around the synchronous ChromaDB client.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import chromadb
import structlog
from chromadb.config import Settings

from kodiak.rag.chunker import Chunk

logger = structlog.get_logger(__name__)

# Data types


@dataclass
class SearchResult:
    chunk_id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str
    name: str | None
    score: float  # cosine similarity [0, 1]
    metadata: dict[str, Any]


@dataclass
class UpsertStats:
    upserted: int
    skipped: int
    collection: str


# Vector Store
class VectorStore:
    """
    Async wrapper around ChromaDB for storing and retrieving code chunk embeddings.

    Each repository gets its own ChromaDB collection named ``repo_{repo_id}``.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        persist_directory: str | None = None,
    ):
        if persist_directory:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(anonymized_telemetry=False),
            )
        self._loop = asyncio.get_event_loop()
        self._collections: dict[str, chromadb.Collection] = {}

    # Collection management

    def _collection_name(self, repo_id: str) -> str:
        # ChromaDB collection names must be 3-63 chars, alphanumeric + underscores
        safe = repo_id.replace("/", "_").replace("-", "_").replace(".", "_")
        return f"repo_{safe}"[:63]

    def _get_or_create_collection(self, repo_id: str) -> chromadb.Collection:
        name = self._collection_name(repo_id)
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    async def _run_sync(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a sync ChromaDB call in the thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(fn, *args, **kwargs))

    # Upsert

    async def upsert_chunks(
        self,
        repo_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> UpsertStats:
        """Store chunks + embeddings into the repo's collection."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )

        if not chunks:
            return UpsertStats(upserted=0, skipped=0, collection=self._collection_name(repo_id))

        collection = self._get_or_create_collection(repo_id)

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "file_path": c.file_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "language": c.language,
                "chunk_type": c.chunk_type.value,
                "name": c.name or "",
                "parent": c.parent or "",
                "token_estimate": c.token_estimate,
                **c.metadata,
            }
            for c in chunks
        ]

        await self._run_sync(
            collection.upsert,
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "vector_store_upsert",
            repo_id=repo_id,
            count=len(chunks),
            collection=collection.name,
        )

        return UpsertStats(
            upserted=len(chunks),
            skipped=0,
            collection=collection.name,
        )

    # Search

    async def search(
        self,
        repo_id: str,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Semantic search within a repo collection.

        Args:
            repo_id: Repository identifier.
            query_embedding: Query vector.
            top_k: Number of results.
            filters: ChromaDB where-clause filters, e.g. ``{"language": "python"}``.
        """
        collection = self._get_or_create_collection(repo_id)

        where = self._build_where(filters) if filters else None

        raw = await self._run_sync(
            collection.query,
            query_embeddings=[query_embedding],
            n_results=min(top_k, await self._run_sync(collection.count) or top_k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results: list[SearchResult] = []
        ids = raw["ids"][0]
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        distances = raw["distances"][0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances, strict=False):
            # ChromaDB cosine distance → similarity
            score = 1.0 - dist
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=doc,
                    file_path=meta.get("file_path", ""),
                    start_line=int(meta.get("start_line", 0)),
                    end_line=int(meta.get("end_line", 0)),
                    language=meta.get("language", ""),
                    chunk_type=meta.get("chunk_type", ""),
                    name=meta.get("name") or None,
                    score=score,
                    metadata=meta,
                )
            )

        return results

    async def search_multi_repo(
        self,
        repo_ids: list[str],
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search across multiple repositories and merge results."""
        tasks = [self.search(repo_id, query_embedding, top_k=top_k) for repo_id in repo_ids]
        per_repo = await asyncio.gather(*tasks, return_exceptions=True)

        merged: list[SearchResult] = []
        for repo_id, result in zip(repo_ids, per_repo, strict=False):
            if isinstance(result, Exception):
                logger.warning("search_repo_error", repo_id=repo_id, error=str(result))
                continue
            merged.extend(result)

        # Sort by score descending, return top_k
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:top_k]

    # Deletion

    async def delete_file(self, repo_id: str, file_path: str) -> int:
        """Delete all chunks belonging to a specific file."""
        collection = self._get_or_create_collection(repo_id)
        # Get IDs matching file_path
        raw = await self._run_sync(
            collection.get,
            where={"file_path": {"$eq": file_path}},
            include=[],
        )
        ids = raw.get("ids", [])
        if ids:
            await self._run_sync(collection.delete, ids=ids)
        logger.info("vector_store_delete_file", repo_id=repo_id, file=file_path, deleted=len(ids))
        return len(ids)

    async def delete_repo(self, repo_id: str) -> None:
        """Drop the entire collection for a repository."""
        name = self._collection_name(repo_id)
        try:
            await self._run_sync(self._client.delete_collection, name)
            self._collections.pop(name, None)
            logger.info("vector_store_delete_repo", repo_id=repo_id)
        except Exception as exc:
            logger.warning("vector_store_delete_repo_failed", repo_id=repo_id, error=str(exc))

    # Stats / Health

    async def count(self, repo_id: str) -> int:
        collection = self._get_or_create_collection(repo_id)
        return await self._run_sync(collection.count)

    async def list_repos(self) -> list[str]:
        collections = await self._run_sync(self._client.list_collections)
        return [c.name for c in collections if c.name.startswith("repo_")]

    async def health_check(self) -> bool:
        try:
            await self._run_sync(self._client.heartbeat)
            return True
        except Exception as exc:
            logger.error("vector_store_health_check_failed", error=str(exc))
            return False

    # Helpers

    def _build_where(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Convert simple key=value filter dict to ChromaDB where clause."""
        if len(filters) == 1:
            k, v = next(iter(filters.items()))
            return {k: {"$eq": v}}
        return {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}
