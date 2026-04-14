import chromadb
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Optional

class RailVectorVault:
    def __init__(self, embedder_instance, db_path="./vector_db", collection_name = "rail_safety", reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2", sync_bm25 = True):
        # We pass the embedder IN. This is called 'Dependency Injection'.
        self.embedder = embedder_instance 
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        self.documents = []
        self.metadatas = []
        self.ids = []
        self.bm25 = None
        self.reranker = CrossEncoder(reranker_model, device='cuda' if torch.cuda.is_available() else 'cpu')
        if sync_bm25:
            self._refresh_search_indices()
    def _refresh_search_indices(self):
        db_data = self.collection.get()
        self.documents = db_data['documents']
        self.metadatas = db_data['metadatas']
        self.ids = db_data['ids']
        
        if self.documents:
            tokenized_corpus = [doc.lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)

        
    def add_documents(self, chunks, metadatas):
        # The Vault asks the Embedder to do its job
        vectors = self.embedder.generate_embeddings(chunks)
        ids = [f"id_{i}" for i in range(len(chunks))]
        
        self.collection.add(
            documents=chunks,
            embeddings=vectors.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
    def query(self, question, n_results=3):
        """
        Method1 : Performs a semantic search.
        1. Embeds the question using the injected embedder.
        2. Queries ChromaDB for the closest matches.
        """
        # Embed the query string
        query_vector = self.embedder.generate_embeddings([question])
        query_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector
        
        # Search the collection
        results = self.collection.query(
            query_embeddings=query_list,
            n_results=n_results
        )
        return results
        
    def hybrid_query(self, question: str, n_results: int = 5, rrf_k: int = 60) -> List[Dict]:
        """Method 2: Hybrid Search (BM25 + Vector) using Reciprocal Rank Fusion."""
        if not self.bm25:
            return self.query(question, n_results=n_results)

        # 1. Vector Search
        vector_res = self.query(question, n_results=n_results * 2)
        vector_ids = vector_res['ids'][0]

        # 2. BM25 Search
        tokenized_query = question.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:n_results * 2]
        bm25_ids = [self.ids[i] for i in top_bm25_indices]

        # 3. Reciprocal Rank Fusion
        rank_scores = {}
        for rank, id_ in enumerate(vector_ids):
            rank_scores[id_] = rank_scores.get(id_, 0) + 1 / (rrf_k + rank + 1)
        for rank, id_ in enumerate(bm25_ids):
            rank_scores[id_] = rank_scores.get(id_, 0) + 1 / (rrf_k + rank + 1)

        # Sort and return combined results with metadata
        sorted_ids = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
        
        final_results = []
        id_to_data = {self.ids[i]: (self.documents[i], self.metadatas[i]) for i in range(len(self.ids))}
        for id_, _ in sorted_ids:
            doc, meta = id_to_data[id_]
            final_results.append({"id": id_, "text": doc, "metadata": meta})
            
        return final_results

    def rerank_query(self, question: str, n_results: int = 5, n_initial: int = 25) -> List[Dict]:
        """Method 3: Hybrid Search + Cross-Encoder Reranking."""
        # Get candidates from Hybrid
        candidates = self.hybrid_query(question, n_results=n_initial)
        if not candidates: return []

        # Prepare pairs for Reranker
        pairs = [[question, c['text']] for c in candidates]
        scores = self.reranker.predict(pairs)

        # Sort by reranker score
        ranked_indices = np.argsort(scores)[::-1][:n_results]
        return [candidates[i] for i in ranked_indices]
