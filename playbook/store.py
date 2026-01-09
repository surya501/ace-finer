"""ChromaDB-backed playbook storage."""

import shutil
from datetime import datetime, timezone
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from playbook.schema import Rule


class PlaybookStore:
    """
    ChromaDB-backed rule storage with embedding search.

    All methods are synchronous (Chroma is sync).
    """

    def __init__(self, path: str = "./playbook_db", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the playbook store.

        Args:
            path: Directory for ChromaDB persistence
            embedding_model: Sentence transformer model for embeddings
        """
        self.path = path
        self.client = chromadb.PersistentClient(path=path)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name="rules",
            embedding_function=self.embed_fn
        )

    def add_rule(self, rule: Rule) -> None:
        """
        Insert a new rule. Skips if rule_id already exists.

        Args:
            rule: Rule to add
        """
        existing = self.collection.get(ids=[rule.rule_id])
        if existing["ids"]:
            return  # Already exists

        self.collection.add(
            ids=[rule.rule_id],
            documents=[rule.content],
            metadatas=[rule.to_metadata()]
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[Rule]:
        """
        Find top-k rules by semantic similarity to query.

        Args:
            query: Text to search for
            top_k: Maximum number of rules to return

        Returns:
            List of Rule objects, most similar first
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count())
        )

        rules = []
        for i, rule_id in enumerate(results["ids"][0]):
            rules.append(Rule.from_query_result(
                rule_id=rule_id,
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i]
            ))
        return rules

    def update_stats(self, rule_id: str, success: bool) -> None:
        """
        Increment success or failure count for a rule.

        Args:
            rule_id: ID of rule to update
            success: True to increment success, False for failure
        """
        existing = self.collection.get(ids=[rule_id])
        if not existing["ids"]:
            return

        metadata = existing["metadatas"][0]
        if success:
            metadata["success_count"] = metadata.get("success_count", 0) + 1
        else:
            metadata["failure_count"] = metadata.get("failure_count", 0) + 1
        metadata["last_used"] = datetime.now(timezone.utc).isoformat()

        self.collection.update(ids=[rule_id], metadatas=[metadata])

    def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding vector for text (for duplicate detection).

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.embed_fn([text])[0]
        # Convert numpy array to list if needed
        if hasattr(embedding, 'tolist'):
            return embedding.tolist()
        return list(embedding)

    def get_all_rules(self) -> list[Rule]:
        """
        Retrieve all rules (for curation).

        Returns:
            List of all Rule objects
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.get()
        rules = []
        for i, rule_id in enumerate(results["ids"]):
            rules.append(Rule.from_query_result(
                rule_id=rule_id,
                content=results["documents"][i],
                metadata=results["metadatas"][i]
            ))
        return rules

    def delete_rule(self, rule_id: str) -> None:
        """
        Delete a rule by ID.

        Args:
            rule_id: ID of rule to delete
        """
        self.collection.delete(ids=[rule_id])

    def count(self) -> int:
        """
        Total number of rules.

        Returns:
            Number of rules in the store
        """
        return self.collection.count()

    def checkpoint(self, name: str) -> None:
        """
        Copy current DB to checkpoint directory.

        Args:
            name: Path for the checkpoint
        """
        # Ensure parent directory exists
        Path(name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.path, name, dirs_exist_ok=True)
