"""
Persistent Skill Store using ChromaDB.

Used to cache LLM-generated domain models and acquired capabilities to 
prevent duplicate execution of the environment probe parsing logic.
"""

from __future__ import annotations

import json
from typing import Any

import chromadb
from loguru import logger


class SkillStore:
    """Manages persistent caching of analyzed domain models using ChromaDB."""

    def __init__(self, path: str = ".chroma"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name="domain_models",
            metadata={"description": "Caches of analyzed domain task classes and their structured sub-problems."}
        )

    def store_domain(self, task_class: str, description: str, domain_model: dict[str, Any]) -> None:
        """Cache exactly the JSON data for a task class."""
        try:
            self.collection.upsert(
                documents=[json.dumps(domain_model)],
                metadatas=[{"task_class": task_class, "description": description}],
                ids=[task_class]
            )
            logger.info(f"SkillStore: Cached domain model for '{task_class}'")
        except Exception as e:
            logger.error(f"SkillStore failed to store {task_class}: {e}")

    def get_domain(self, task_class: str) -> dict[str, Any] | None:
        """Retrieve a cached domain exact match by task_class."""
        try:
            result = self.collection.get(ids=[task_class])
            if result and result.get("documents") and len(result["documents"]) > 0:
                doc = result["documents"][0]
                logger.info(f"SkillStore: CACHE HIT for '{task_class}'")
                return json.loads(doc)
        except Exception as e:
            logger.warning(f"SkillStore fetch failed for {task_class}: {e}")

        logger.info(f"SkillStore: CACHE MISS for '{task_class}'")
        return None
