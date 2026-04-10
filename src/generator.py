import json
import random
import requests
import time
import os

class RailDataGenerator:
    def __init__(self, vault_instance, api_url, api_key):
        self.vault = vault_instance
        self.api_url = api_url #"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
        self.api_key = api_key

    def _call_teacher(self, system_prompt, user_query):
        payload = {
            "contents": [{"parts": [{"text": user_query}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]}
        }
        
        # Increased backoff steps to handle stricter 2026 rate limits
        for delay in [2, 4, 8, 16, 32]:
            try:
                # Added a 30s timeout to prevent the script from hanging forever
                response = requests.post(
                    f"{self.api_url}?key={self.api_key}", 
                    json=payload,
                    timeout=30 
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "")
                elif response.status_code == 429:
                    print(f"Rate limit hit. Retrying in {delay}s...")
                else:
                    print(f"Teacher API Error {response.status_code}: {response.text}")
            except requests.exceptions.Timeout:
                print("Request timed out. Retrying...")
            except Exception as e:
                print(f"Request Exception: {e}")
            
            time.sleep(delay)
        return None

    def generate_training_sample(self, chunk_text, file_name, page_num):
        """
        1. Grab a random chunk from the vault.
        2. Ask Teacher to generate a complex question + reasoning process.
        """


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

    def create_dataset(self, num_samples=100, output_path="data/training/rail_dataset.jsonl"):
        samples = []
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Starting generation of {num_samples} sample(s)...")
        # Get random chunks from collection
        all_data = self.vault.collection.get()
        if not all_data or not all_data['documents']:
            return None
        total_chunks = len(all_data['documents'])
        # idx = random.randint(0, len(all_data['documents']) - 1)
        actual_sample_count = min(num_samples, total_chunks)
        indices = random.sample(range(total_chunks), actual_sample_count)
        if num_samples >= total_chunks:
            print(f"only {total_chunks} sample(s) available, so creating {actual_sample_count} unique sample(s)")

        for idx in indices::
            chunk_text = all_data['documents'][idx]
            file_name = all_data['metadatas'][idx].get('source', 'Unknown')
            page_num = all_data['metadatas'][idx].get('page', '?')
            sample = self.generate_training_sample(chunk_text, file_name, page_number)
            if sample:
                samples.append(sample)
                # Append to file immediately so you don't lose data if it crashes
                with open(output_path, 'a') as f:
                    f.write(json.dumps(sample) + "\n")
                print(f"Generated {i+1}/{num_samples}")
            
            # MANDATORY COOL-DOWN: 
            # 3 seconds between requests helps stay under the 20 RPM limit
            time.sleep(3) 
            
        return output_path
