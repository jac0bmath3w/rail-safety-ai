import json
import pandas as pd
from tqdm import tqdm
from typing import List

class RailRetrieverEvaluator:
    def __init__(self, vault):
        self.vault = vault

    def evaluate(self, eval_set_path: str, batch_size: int = 16, n_initial = 30):
        """
        Evaluates Vector, Hybrid, and Rerank methods against the Golden Set.
        """
        with open(eval_set_path, 'r') as f:
            eval_data = json.load(f)
        questions = [item['question'] for item in eval_data]
        target_ids = [item['answer_chunk_id'] for item in eval_data]
        
        metrics = {
            "Vector": {"hit@1": 0, "hit@3": 0, "hit@5": 0, "mrr": 0},
            "Hybrid": {"hit@1": 0, "hit@3": 0, "hit@5": 0, "mrr": 0},
            "Rerank": {"hit@1": 0, "hit@3": 0, "hit@5": 0, "mrr": 0}
        }
        for i in tqdm(range(0, len(questions), batch_size), desc="Benchmarking in Batches"):
            
            # Slice our questions and targets for this specific batch
            batch_qs = questions[i : i + batch_size]
            batch_targets = target_ids[i : i + batch_size]

            batch_vec_results = self.vault.query(batch_qs, n_results=10)
            batch_hyb_results = self.vault.hybrid_query(batch_qs, n_results=10)
            batch_rerank_results = self.vault.rerank_query(batch_qs, n_results=5, n_initial=n_initial)

            for j in range(len(batch_qs)):
                # Vector Metrics
                vec_ids = [r['id'] for r in batch_vec_results[j]]
                self._update_metrics(metrics["Vector"], vec_ids, batch_targets[j])

                # Hybrid Metrics
                hyb_ids = [r['id'] for r in batch_hyb_results[j]]
                self._update_metrics(metrics["Hybrid"], hyb_ids, batch_targets[j])

                # Rerank Metrics
                rerank_ids = [r['id'] for r in batch_rerank_results[j]]
                self._update_metrics(metrics["Rerank"], rerank_ids, batch_targets[j])

        total_q = len(questions)
        for method in metrics:
            for key in metrics[method]:
                metrics[method][key] /= total_q


        return pd.DataFrame(metrics).T


    def _update_metrics(self, metric_dict, retrieved_ids, target_id):
        if target_id in retrieved_ids:
            rank = retrieved_ids.index(target_id) + 1
            metric_dict["mrr"] += 1 / rank
            if rank <= 1: metric_dict["hit@1"] += 1
            if rank <= 3: metric_dict["hit@3"] += 1
            if rank <= 5: metric_dict["hit@5"] += 1
