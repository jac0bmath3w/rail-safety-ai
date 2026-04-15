import chromadb
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Optional

class RailVectorVault:
    def __init__(self, embedder_instance, db_path="./vector_db", collection_name = "rail_safety", reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2", sync_bm25 = True):#, rerank_instruction: str = "Query: "):
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
        
    def query(self, questions: List[str], n_results: int = 5, where: Optional[Dict] = None) -> List[Dict]:
        """
        Method1 : Performs a semantic search.
        1. Embeds the question using the injected embedder.
        2. Queries ChromaDB for the closest matches.
        """
        # Embed the query string
        query_vector = self.embedder.generate_embeddings(questions)
        query_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector
        
        # Search the collection
        res = self.collection.query(
            query_embeddings=query_list,
            n_results=n_results,
            where = where
        )

        # formatted = []
        # if res['documents']:
        #     for i in range(len(res['documents'][0])):
        #         formatted.append({
        #             "id": res['ids'][0][i],
        #             "text": res['documents'][0][i],
        #             "metadata":res['metadatas'][0][i]
        #         })
        # return formatted
        formatted_batch = []
        for q_idx in range(len(res['documents'])):
            question_results = []
            for i in range(len(res['documents'][q_idx])):
                question_results.append({
                    "id": res['ids'][q_idx][i],
                    "text": res['documents'][q_idx][i],
                    "metadata": res['metadatas'][q_idx][i]
                })
            formatted_batch.append(question_results)
        return formatted_batch
        
    def hybrid_query(self, questions: List[str], n_results: int = 5, where: Optional[Dict] = None, rrf_k: int = 60) -> List[Dict]:
        """Method 2: Hybrid Search (BM25 + Vector) using Reciprocal Rank Fusion."""
        if not self.bm25:
            return self.query(questions, n_results=n_results)

        # 1. Vector Search
        all_vector_res = self.query(questions, n_results=n_results * 5, where=where)
        final_batch_results = []
        # vector_res = self.query(question, n_results=n_results * 5, where = where)
        # vector_ids = [r['id'] for r in vector_res]
        # vector_ids = vector_res['ids'][0]

        # 2. BM25 Search
        for q_idx, question in enumerate(questions):
            vector_ids = [r['id'] for r in all_vector_res[q_idx]]
            
            tokenized_query = question.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)

            # Filter BM25 by metadata if 'where' is provided
            rank_scores = {}
        
            # Map indices back to IDs for the BM25 loop
            # for i, (id_, meta) in enumerate(zip(self.ids, self.metadatas)):
            #     # Simple metadata filter for 'source' if provided
            #     if where and 'source' in where:
            #         if meta.get('source') != where['source']:
            #             continue
            
                # Reciprocal Rank for BM25 (using score-based sorting for rank)
                # This is a simplified version of RRF integration


            # rank_scores = {}
            # for rank, id_ in enumerate(vector_ids):
            #     rank_scores[id_] = rank_scores.get(id_, 0) + 1 / (rrf_k + rank + 1)
            # for rank, id_ in enumerate(bm25_ids):
            #     rank_scores[id_] = rank_scores.get(id_, 0) + 1 / (rrf_k + rank + 1)
    
            # RRF Implementation
            
            bm25_top_indices = np.argsort(bm25_scores)[::-1]
            bm25_count = 0
            for rank, idx in enumerate(bm25_top_indices):
                # if bm25_scores[idx] <= 0:
                #     break
                id_ = self.ids[idx]
                meta = self.metadatas[idx]
                
                # Filter
                if where and 'source' in where and meta.get('source') != where['source']:
                    continue
                    
                rank_scores[id_] = rank_scores.get(id_, 0) + 1 / (rrf_k + bm25_count + 1)
                bm25_count += 1
                if bm25_count >= n_results * 5: break
            
            vector_count = 0
            id_to_meta = {self.ids[i]: self.metadatas[i] for i in range(len(self.ids))}
            for rank, id_ in enumerate(vector_ids):
                meta = id_to_meta.get(id_, {})
                # Filter
                if where and 'source' in where and meta.get('source') != where['source']:
                    continue
                rank_scores[id_] = rank_scores.get(id_, 0) + 1 / (rrf_k + vector_count + 1)
                vector_count+=1
    
            sorted_ids = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
            
            # Hydrate results
            id_to_data = {self.ids[i]: (self.documents[i], self.metadatas[i]) for i in range(len(self.ids))}
            final_batch_results.append([
                                            {"id": k, "text": id_to_data[k][0], "metadata": id_to_data[k][1]} 
                                            for k, v in sorted_ids
                                            ])
        return final_batch_results

        # top_bm25_indices = np.argsort(bm25_scores)[::-1][:n_results * 2]
        # bm25_ids = [self.ids[i] for i in top_bm25_indices]

        # # 3. Reciprocal Rank Fusion
        # rank_scores = {}
        # for rank, id_ in enumerate(vector_ids):
        #     rank_scores[id_] = rank_scores.get(id_, 0) + 1 / (rrf_k + rank + 1)
        # for rank, id_ in enumerate(bm25_ids):
        #     rank_scores[id_] = rank_scores.get(id_, 0) + 1 / (rrf_k + rank + 1)

        # # Sort and return combined results with metadata
        # sorted_ids = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
        
        # final_results = []
        # id_to_data = {self.ids[i]: (self.documents[i], self.metadatas[i]) for i in range(len(self.ids))}
        # for id_, _ in sorted_ids:
        #     doc, meta = id_to_data[id_]
        #     final_results.append({"id": id_, "text": doc, "metadata": meta})
            
        # return final_results

    def rerank_query(self, questions: List[str], n_results: int = 5, n_initial: int = 25, where: Optional[Dict] = None) -> List[Dict]:
        """Method 3: Hybrid Search + Cross-Encoder Reranking."""
        # Get candidates from Hybrid
        candidates_batch = self.hybrid_query(questions, n_results=n_initial, where = where)
        # if not candidates: return []

        # Prepare pairs for Reranker
        # pairs = [[question, c['text']] for c in candidates]

        # if "bge" in self.reranker_model.lower():
        #     pairs = [[f"Query: {question}", f"Passage: {c['text']}"] for c in candidates]
        # else:
        #     pairs = [[question, c['text']] for c in candidates]

        all_pairs = []
        for q_idx, question in enumerate(questions):
            for c in candidates_batch[q_idx]:
                if "bge" in str(self.reranker).lower():
                    all_pairs.append([f"Query: {question}", f"Passage: {c['text']}"])
                else:
                    all_pairs.append([question, c['text']])
                    
        # scores = self.reranker.predict(pairs)
        all_scores = self.reranker.predict(all_pairs, batch_size=32)
        final_batch_results = []
        score_idx = 0
        for q_idx in range(len(questions)):
            num_cands = len(candidates_batch[q_idx])
            q_scores = all_scores[score_idx : score_idx + num_cands]
            score_idx += num_cands
            
            ranked_indices = np.argsort(q_scores)[::-1][:n_results]
            final_batch_results.append([candidates_batch[q_idx][i] for i in ranked_indices])
            
        return final_batch_results
