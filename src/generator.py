import json
import random
import requests
import time

class RailDataGenerator:
    """
    Teacher model interface to generate synthetic training data for the Student (Llama 3.2 3B).
    Uses Gemini 2.5 Flash to create high-quality Reasoning/Answer pairs from manual chunks.
    """
    def __init__(self, vault_instance, api_url, api_key):
        self.vault = vault_instance
        self.api_url = api_url  # "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
        self.api_key = api_key  # Key provided by environment at runtime

    def _call_teacher(self, system_prompt, user_query):
        """Helper to call the Gemini API with exponential backoff."""
        payload = {
            "contents": [{"parts": [{"text": user_query}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]}
        }
        
        for delay in [1, 2, 4, 8, 16]:
            try:
                response = requests.post(f"{self.api_url}?key={self.api_key}", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "")
            except Exception:
                pass
            time.sleep(delay)
        return None

    def generate_training_sample(self):
        """
        1. Grab a random chunk from the vault.
        2. Ask Teacher to generate a complex question + reasoning process.
        """
        # Get random chunks from collection
        all_data = self.vault.collection.get()
        if not all_data or not all_data['documents']:
            return None
            
        idx = random.randint(0, len(all_data['documents']) - 1)
        chunk_text = all_data['documents'][idx]
        file_name = all_data['metadatas'][idx].get('source', 'Unknown')
        page_num = all_data['metadatas'][idx].get('page', '?')

        system_prompt = (
            "You are a Senior FRA Rail Safety Expert. Your task is to generate training data "
            "for a student model. Based on the provided manual excerpt, create a challenging "
            "technical question and a perfect response following the 4-Phase Thinking Process.\n\n"
            "PHASE 1: CONTEXTUAL AUDIT\nPHASE 2: EVIDENCE MAPPING\nPHASE 3: SYNTHESIS\nPHASE 4: VERIFICATION\n\n"
            "Output MUST be in valid JSON format: "
            "{'question': '...', 'thinking': '...', 'answer': '...'}"
        )

        user_query = f"MANUAL EXCERPT ({file_name}, Page {page_num}):\n{chunk_text}"
        
        raw_output = self._call_teacher(system_prompt, user_query)
        if not raw_output:
            return None

        # Clean JSON if model included markdown blocks
        clean_json = raw_output.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean_json)
        except:
            return None

    def create_dataset(self, num_samples=50, output_path="data/training/rail_dataset.jsonl"):
        """Generates a full dataset and saves to disk."""
        samples = []
        print(f"Generating {num_samples} synthetic samples...")
        
        for i in range(num_samples):
            sample = self.generate_training_sample()
            if sample:
                samples.append(sample)
                print(f"Generated sample {i+1}/{num_samples}")
            
        with open(output_path, 'w') as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        
        return output_path
