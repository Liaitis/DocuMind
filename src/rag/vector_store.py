"""
Vector Store using ChromaDB with persistent storage.
Migrated to the new ChromaDB PersistentClient.
"""

import logging
from pathlib import Path
from typing import Optional, List
import os

import chromadb

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store using ChromaDB with persistent storage."""
    
    def __init__(self, persist_dir: Optional[str] = None):
        """
        Initialize vector store with ChromaDB PersistentClient.
        
        Args:
            persist_dir: Directory to persist the database.
                        Defaults to 'data/chroma' if None.
        """
        logger.info("Initializing VectorStore...")
        
        if persist_dir is None:
            persist_dir = os.path.join("data", "chroma")
        
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Persist directory: {persist_dir}")
        
        self.persist_dir = persist_dir
        self.documents = []  # fallback in-memory list
        self.client = None
        self.collection = None
        self.is_in_memory = False
        
        try:
            logger.info("Attempting to initialize ChromaDB PersistentClient...")
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("✓ ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            logger.warning("Falling back to in-memory storage...")
            self.documents = []
            self.is_in_memory = True
    
    def add_documents(self, documents: List[str], metadata: Optional[List[dict]] = None):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document strings
            metadata: Optional list of metadata dicts
        """
        if not documents:
            return
        
        if self.collection is not None:
            try:
                ids = [f"doc_{i}" for i in range(self.collection.count(), self.collection.count() + len(documents))]
                if metadata is None:
                    metadata = [{"source": "uploaded"} for _ in documents]
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadata
                )
                logger.info(f"✓ Added {len(documents)} documents to ChromaDB")
            except Exception as e:
                logger.error(f"Error adding documents to ChromaDB: {e}")
                # Fallback to in-memory
                self.documents.extend(documents)
                self.is_in_memory = True
        else:
            self.documents.extend(documents)
            logger.info(f"✓ Added {len(documents)} documents to in-memory store")
    
    def search(self, query: str, limit: int = 5) -> List[dict]:
        """
        Search for similar documents.
        
        Returns:
            List of dicts with 'content', 'distance', 'metadata'
        """
        if self.collection is not None:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=limit
                )
                documents = []
                if results and results['documents'] and len(results['documents']) > 0:
                    for i, doc in enumerate(results['documents'][0]):
                        documents.append({
                            'content': doc,
                            'distance': results['distances'][0][i] if results['distances'] else 0,
                            'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                        })
                return documents
            except Exception as e:
                logger.error(f"ChromaDB search error: {e}")
                # Fallback to simple in-memory search
                return self._in_memory_search(query, limit)
        else:
            return self._in_memory_search(query, limit)
    
    def _in_memory_search(self, query: str, limit: int) -> List[dict]:
        """Simple in-memory search (fallback)."""
        results = []
        query_lower = query.lower()
        for doc in self.documents:
            if query_lower in doc.lower():
                results.append({'content': doc, 'distance': 0, 'metadata': {}})
        return results[:limit]
    
    def get_count(self) -> int:
        """Get total number of documents."""
        if self.collection is not None:
            try:
                return self.collection.count()
            except:
                return len(self.documents)
        return len(self.documents)
    
    def clear(self):
        """Clear all documents."""
        if self.collection is not None:
            try:
                all_ids = self.collection.get()['ids']
                if all_ids:
                    self.collection.delete(ids=all_ids)
                logger.info("Cleared ChromaDB collection")
            except Exception as e:
                logger.error(f"Error clearing ChromaDB: {e}")
        self.documents = []