import json
import pandas as pd
from tqdm import tqdm
from typing import List

class RailRetrieverEvaluator:
    def __init__(self, vault):
        self.vault = vault

    def evaluate(self, eval_set_path: str):
        """
        Evaluates Vector, Hybrid, and Rerank methods against the Golden Set.
        """
        with open(eval_set_path, 'r') as f:
            eval_data = json.load(f)

        metrics = {
            "Vector": {"hit@1": 0, "hit@3": 0, "hit@5": 0, "mrr": 0},
            "Hybrid": {"hit@1": 0, "hit@3": 0, "hit@5": 0, "mrr": 0},
            "Rerank": {"hit@1": 0, "hit@3": 0, "hit@5": 0, "mrr": 0}
        }

        for item in tqdm(eval_data, desc="Benchmarking Retriever"):
            query = item['question']
            target_id = item['answer_chunk_id']

            # 1. Test Vector
            vec_res = self.vault.query(query, n_results=10)
            vec_ids = [r['id'] for r in vec_res]
            self._update_metrics(metrics["Vector"], vec_ids, target_id)

            # 2. Test Hybrid
            hyb_res = self.vault.hybrid_query(query, n_results=10)
            hyb_ids = [r['id'] for r in hyb_res]
            self._update_metrics(metrics["Hybrid"], hyb_ids, target_id)

            # 3. Test Rerank
            rerank_res = self.vault.rerank_query(query, n_results=5, n_initial=25)
            rerank_ids = [r['id'] for r in rerank_res]
            self._update_metrics(metrics["Rerank"], rerank_ids, target_id)

        # Average out the metrics
        n = len(eval_data)
        for method in metrics:
            for key in metrics[method]:
                metrics[method][key] /= n

        return pd.DataFrame(metrics).T

    def _update_metrics(self, metric_dict, retrieved_ids, target_id):
        if target_id in retrieved_ids:
            rank = retrieved_ids.index(target_id) + 1
            metric_dict["mrr"] += 1 / rank
            if rank <= 1: metric_dict["hit@1"] += 1
            if rank <= 3: metric_dict["hit@3"] += 1
            if rank <= 5: metric_dict["hit@5"] += 1
