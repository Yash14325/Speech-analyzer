# rag/retriever.py
"""
RAG Retriever - Uses ChromaDB for retrieval.
ChromaDB uses its own built-in embeddings for fast local retrieval.
"""

from typing import Dict, List, Optional

try:
    import chromadb

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("ChromaDB not installed. Install with: pip install chromadb")

from rag.config import COLLECTION_NAME, TOP_K_RESULTS
from rag.knowledge_base import KnowledgeBase


class RAGRetriever:
    """
    Retrieval Augmented Generation system using ChromaDB when available,
    with keyword fallback for environments where ChromaDB cannot initialize.
    """

    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.collection = None
        self._initialized = False
        self._setup_vector_store()

    def _setup_vector_store(self):
        """Setup ChromaDB vector store with built-in embeddings."""
        if not CHROMA_AVAILABLE:
            print("ChromaDB not available, using fallback retrieval")
            return

        try:
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Speech analysis knowledge base"},
            )

            if self.collection.count() == 0:
                self._index_documents()

            self._initialized = True
            print(f"ChromaDB initialized with {self.collection.count()} documents")
        except Exception as e:
            print(f"ChromaDB setup failed: {e}")
            self._initialized = False

    def _index_documents(self):
        """Index knowledge base documents into ChromaDB."""
        if self.collection is None:
            return

        documents = self.knowledge_base.get_all_documents()
        self.collection.add(
            documents=[doc["content"] for doc in documents],
            ids=[doc["id"] for doc in documents],
            metadatas=[{"category": doc["category"]} for doc in documents],
        )
        print(f"Indexed {len(documents)} knowledge documents")

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        category_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        if not self._initialized or self.collection is None:
            return self._fallback_retrieve(query, top_k, category_filter)

        try:
            where_filter = {"category": category_filter} if category_filter else None
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
            )

            retrieved_docs = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    retrieved_docs.append(
                        {
                            "content": doc,
                            "id": results["ids"][0][i] if results["ids"] else None,
                            "category": (
                                results["metadatas"][0][i].get("category")
                                if results["metadatas"]
                                else None
                            ),
                            "distance": (
                                results["distances"][0][i]
                                if results.get("distances")
                                else None
                            ),
                        }
                    )

            return retrieved_docs
        except Exception as e:
            print(f"Retrieval error: {e}")
            return self._fallback_retrieve(query, top_k, category_filter)

    def _fallback_retrieve(
        self,
        query: str,
        top_k: int,
        category_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Fallback keyword-based retrieval when vector store is unavailable."""
        documents = self.knowledge_base.get_all_documents()
        if category_filter:
            documents = [doc for doc in documents if doc["category"] == category_filter]

        query_words = query.lower().split()
        scored_docs = []

        for doc in documents:
            content_lower = doc["content"].lower()
            score = sum(1 for word in query_words if word in content_lower)
            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "content": doc["content"],
                "id": doc["id"],
                "category": doc["category"],
                "distance": None,
            }
            for _, doc in scored_docs[:top_k]
        ]

    def get_context_for_analysis(self, analysis_type: str, metrics: Dict) -> str:
        """Get relevant context for a specific type of analysis."""
        query_parts = []

        if analysis_type == "communication":
            query_parts.append("speech communication clarity fluency vocabulary structure")
            if metrics.get("speech_rate"):
                rate = metrics["speech_rate"]
                if isinstance(rate, (int, float)):
                    if rate < 120:
                        query_parts.append("slow speech rate thoughtful delivery")
                    elif rate > 160:
                        query_parts.append("fast speech rate rapid delivery")
                    else:
                        query_parts.append("normal speech rate optimal pacing")
                query_parts.append(f"{rate} words per minute WPM")
            if metrics.get("pause_ratio"):
                pause = metrics["pause_ratio"]
                if isinstance(pause, (int, float)):
                    if pause > 0.25:
                        query_parts.append("high pause ratio hesitation disfluency")
                    else:
                        query_parts.append("good fluency smooth delivery")
            if metrics.get("transcript_preview"):
                query_parts.append("transcript analysis vocabulary assessment")

        elif analysis_type == "confidence":
            query_parts.append("confidence vocal delivery emotional tone")
            if metrics.get("energy_level"):
                energy = str(metrics["energy_level"]).lower()
                query_parts.append(f"{energy} energy level projection")
            if metrics.get("pitch_variance"):
                pitch = metrics["pitch_variance"]
                if isinstance(pitch, (int, float)):
                    if pitch > 40:
                        query_parts.append("high pitch variance nervous emotional")
                    elif pitch < 10:
                        query_parts.append("low pitch variance monotone")
                    else:
                        query_parts.append("normal pitch variance engaged speaking")
            if metrics.get("pause_ratio"):
                pause = metrics["pause_ratio"]
                if isinstance(pause, (int, float)):
                    if pause > 0.25:
                        query_parts.append("hesitation pauses uncertainty")
                    else:
                        query_parts.append("confident pauses strategic")

        elif analysis_type == "personality":
            query_parts.append("personality traits communication style behavioral indicators")
            if metrics.get("fluency_level"):
                query_parts.append(f"{metrics['fluency_level']} fluency expressiveness")
            if metrics.get("confidence_level"):
                query_parts.append(f"{metrics['confidence_level']} confidence assertiveness")
            if metrics.get("emotion"):
                query_parts.append(f"{metrics['emotion']} emotional state")

        elif analysis_type == "improvement":
            query_parts.append("improvement tips techniques recommendations")
            if metrics.get("weak_areas"):
                for area in metrics["weak_areas"]:
                    query_parts.append(f"improve {area}")
            else:
                query_parts.append("speaking skills confidence building clarity enhancement")
        else:
            query_parts.append("communication analysis speech evaluation")

        query = " ".join(query_parts)
        docs = self.retrieve(query, top_k=TOP_K_RESULTS, category_filter=analysis_type)

        if not docs:
            docs = self.retrieve(query, top_k=TOP_K_RESULTS)

        if docs:
            context_parts = [f"**Expert Knowledge ({analysis_type.title()}):**\n"]
            for i, doc in enumerate(docs, 1):
                context_parts.append(f"{i}. {doc['content'].strip()}\n")
            return "\n".join(context_parts)

        return ""


_retriever_instance = None


def get_retriever() -> RAGRetriever:
    """Get or create the RAG retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever()
    return _retriever_instance
