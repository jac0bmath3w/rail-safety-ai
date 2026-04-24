import json
import pandas as pd
from tqdm import tqdm
import torch
import requests
import time
from collections import defaultdict
import os

class RailAuditJudge:
    def __init__(self, audit_function, model, tokenizer, vault, api_key="", judge_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"):
        """
        RailAuditJudge: Benchmarks the Auditor's reasoning quality using a "Teacher" LLM as a Judge.
        """
        self.audit_func = audit_function
        self.vault = vault
        self.api_key = api_key
        self.judge_url = judge_url
        self.model = model
        self.tokenizer = tokenizer

    def generate_judge_prompt(self, question, thinking_process, answer, ground_truth):
        """Creates a professional rubric for the Judge LLM."""
        return f"""
        You are an expert FRA Safety Auditor. Grade the following AI-generated safety response against the Ground Truth reference.
        
        [STIMULUS]
        Question: {question}
        Ground Truth Reference (The True Manual Text): {ground_truth}
        
        [AI RESPONSE]
        Thinking Process (What the AI considered): {thinking_process}
        Final Answer: {answer}
        
        [RUBRIC]
        1. FAITHFULNESS (1-5): Is the answer derived ONLY from the context provided in the thinking process? (1 = Hallucinated/Used external knowledge, 5 = Perfectly Grounded)
        2. REGULATORY ACCURACY (1-5): Compare the AI Answer to the Ground Truth Reference. Does the logic match? (1 = Dangerous/Incorrect, 5 = Expert accuracy)
        3. CITATION QUALITY (1-5): Did the model cite specific Pages/Sections correctly within the final answer as per the Thinking Process?
        
        Provide your critique and scores in a structured JSON format.
        """

    def get_judgment(self, judge_prompt):
        """Calls the Judge LLM via API with exponential backoff and structured JSON output."""
        payload = {
            "contents": [{"parts": [{"text": judge_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "faithfulness": {"type": "NUMBER"},
                        "accuracy": {"type": "NUMBER"},
                        "citation": {"type": "NUMBER"},
                        "critique": {"type": "STRING"}
                    },
                    "required": ["faithfulness", "accuracy", "citation", "critique"]
                }
            }
        }

        for delay in [1, 2, 4, 8, 16]:
            try:
                response = requests.post(f"{self.judge_url}?key={self.api_key}", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    text_content = result['candidates'][0]['content']['parts'][0]['text']
                    return json.loads(text_content)
            except Exception:
                time.sleep(delay)
        
        return {"faithfulness": 0, "accuracy": 0, "citation": 0, "critique": "API Failure"}

    def run_benchmark(self, eval_set_path, num_samples=100, batch_size=4, use_dynamic_filter=True, save_path = None):
        """
        Runs E2E evaluation. If use_dynamic_filter=True, it automatically uses the 
        correct source file for every question based on the JSON metadata.
        """
        with open(eval_set_path, 'r') as f:
            eval_data = json.load(f)[:num_samples]
            
        # Group samples by their source file for efficient batching
        grouped_data = defaultdict(list)
        for item in eval_data:
            source_file = item.get('file', 'Unknown') if use_dynamic_filter else None
            grouped_data[source_file].append(item)

        results = []
        
        if save_path is not None:
            os.makedirs(save_path, exist_ok = True)

        
        for source_file, items in grouped_data.items():
            print(f"\n Evaluating {len(items)} samples for source: {source_file or 'ALL (Unfiltered)'}")
            results_source = []
            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                questions = [item['question'] for item in batch]
                
                # Inference with the specific source_filter for this manual
                batch_responses = self.audit_func(
                    questions, 
                    self.vault, 
                    self.tokenizer,
                    self.model, 
                    method='rerank', 
                    n_results=5,
                    source_filter=source_file, 
                    show_context=False 
                )
                
                for idx, raw_output in enumerate(batch_responses):
                    clean_output = raw_output.split("assistant\n\n")[-1]
                    item = batch[idx]
                    ground_truth = item.get('answer_chunk', "N/A")
                    clean_output = raw_output.split("assistant\n\n")[-1] 
    
                    if "[ANSWER]" in clean_output:
                        parts = clean_output.split("[ANSWER]")
                        # Strip the header and any leading/trailing whitespace
                        thinking = parts[0].replace("[THINKING PROCESS]", "").strip()
                        answer = parts[-1].strip()
                    else:
                        thinking, answer = "Parse Error: No [ANSWER] tag", clean_output
                    
                    # try:
                    #     parts = raw_output.split("[ANSWER]")
                    #     thinking = parts[0].replace("[THINKING PROCESS]", "").strip()
                    #     answer = parts[-1].strip()
                    # except Exception:
                    #     thinking, answer = "Parse Error", raw_output

                    judgment = self.get_judgment(self.generate_judge_prompt(item['question'], thinking, answer, ground_truth))

                    results.append({
                        "question": item['question'],
                        "source_file": source_file,
                        "ground_truth": ground_truth, 
                        "ai_thinking": thinking,
                        "ai_answer": answer,
                        "faithfulness": judgment['faithfulness'],
                        "accuracy": judgment['accuracy'],
                        "citation": judgment['citation'],
                        "critique": judgment['critique'],
                        "thinking": thinking
                    })
                    results_source.append({
                        "question": item['question'],
                        "source_file": source_file,
                        "ground_truth": ground_truth, 
                        "ai_thinking": thinking,
                        "ai_answer": answer,
                        "faithfulness": judgment['faithfulness'],
                        "accuracy": judgment['accuracy'],
                        "citation": judgment['citation'],
                        "critique": judgment['critique'],
                        "thinking": thinking
                    })
                
                torch.cuda.empty_cache()
                print(f"   - Batch {i//batch_size + 1}/{(len(items)-1)//batch_size + 1} complete.")
            if save_path and results_source:
                if source_file:
                    # Remove extension from filename for the CSV name
                    name = os.path.splitext(os.path.basename(source_file))[0]
                else:
                    name = 'all'
                
                csv_filename = os.path.join(save_path, f'evaluation_samples_from_source_{name}.csv')
                pd.DataFrame(results_source).to_csv(csv_filename, index=False)
                print(f" >> Saved checkpoint to {csv_filename}")
            # pd.DataFrame(results_source).to_csv(f'{save_path}/evaluation_samples_from_source_{name}.csv', index = False)
            
        return pd.DataFrame(results)

# --- HOW TO RUN ---
# evaluator = RailAuditJudge(run_integrated_audit, vault, api_key="")
# report = evaluator.run_benchmark("retriever_eval_set.json", num_samples=100, use_dynamic_filter=True)
