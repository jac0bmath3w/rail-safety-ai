import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import torch

class RailRetrieverPro:
    def __init__(self, vault, use_reranker=True):
        self.vault = vault
        self.documents = vault.collection.get()['documents']
        self.metadatas = vault.collection.get()['metadatas']
        self.ids = vault.collection.get()['ids']
        
        # 1. Initialize BM25 for Keyword Search
        tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Initialize Cross-Encoder for Re-ranking
        # This model is small, fast, and excellent at 'Logic' matching
        if use_reranker:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
        else:
            self.reranker = None

    def hybrid_query(self, query, n_initial=25):
        """
        Combines BM25 and Vector scores using Reciprocal Rank Fusion (RRF).
        """
        # A. Vector Search (Semantic)
        vector_results = self.vault.query(query, n_results=n_initial)
        vector_ids = vector_results['ids'][0]
        
        # B. BM25 Search (Keyword)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:n_initial]
        
        # C. Simple RRF or Union
        # For now, let's take a Union of both to ensure we don't miss anything
        candidate_indices = set(top_bm25_indices)
        id_to_idx = {id_: i for i, id_ in enumerate(self.ids)}
        for vid in vector_ids:
            if vid in id_to_idx:
                candidate_indices.add(id_to_idx[vid])
                
        return list(candidate_indices)

    def search(self, query, top_k=5, n_initial=30):
        """
        Full Pipeline: Hybrid Retrieval -> Cross-Encoder Re-ranking
        """
        # 1. Hybrid Retrieval
        candidate_indices = self.hybrid_query(query, n_initial=n_initial)
        
        candidate_docs = [self.documents[i] for i in candidate_indices]
        candidate_metas = [self.metadatas[i] for i in candidate_indices]
        
        if not self.reranker:
            return candidate_docs[:top_k], candidate_metas[:top_k]

        # 2. Re-ranking
        # We ask the Cross-Encoder to score (Query, Document) pairs
        pairs = [[query, doc] for doc in candidate_docs]
        scores = self.reranker.predict(pairs)
        
        # 3. Sort by Re-ranker Score
        ranked_indices = np.argsort(scores)[::-1]
        
        final_docs = [candidate_docs[i] for i in ranked_indices[:top_k]]
        final_metas = [candidate_metas[i] for i in ranked_indices[:top_k]]
        
        return final_docs, final_metas

class RetrievalEvaluator:
    def __init__(self, retriever):
        self.retriever = retriever
        # Ground Truth: { "Question": "Target_Chunk_ID" }
        # You'll build this as you find 'Golden' answers
        self.golden_set = {
            "leaking LPG tank car movement permit": "id_of_174_50_chunk", 
            "unaccompanied signal inspection": "id_1493"
        }

    def evaluate(self, k_values=[1, 3, 5]):
        """
        Calculates Hit@K and MRR for the Golden Set.
        """
        results = {f"Hit@{k}": 0 for k in k_values}
        mrr = 0
        
        for query, target_id in self.golden_set.items():
            # Run the search
            docs, metas = self.retriever.search(query, top_k=max(k_values))
            
            # Check if target is in results (This assumes ID is in metadata)
            # You might need to add 'id' to your metadata during ingestion
            found = False
            for rank, meta in enumerate(metas):
                if meta.get('chunk_id') == target_id:
                    # Update Hit@K
                    for k in k_values:
                        if rank < k:
                            results[f"Hit@{k}"] += 1
                    # Update MRR
                    mrr += 1 / (rank + 1)
                    found = True
                    break
        
        # Average results
        num_q = len(self.golden_set)
        for k in results:
            results[k] /= num_q
        
        return {**results, "MRR": mrr / num_q}
