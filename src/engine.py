from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class RailSafetyEngine:
    def __init__(self, model_id="unsloth/Llama-3.2-3B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load the model in 4-bit for memory efficiency (Staff Level Optimization)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            device_map="auto"
        )

    def generate_answer(self, question, context_chunks):
        """
        Takes a question and a list of formatted strings (including metadata).
        """
        # Step 1: Combine context
        context_text = "\n".join(context_chunks)
        
        # Step 2: The "Self-Correcting" Staff Prompt
        prompt = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a BNSF Staff Safety Engineer. 
        INSTRUCTIONS:
        1. Correct any obvious typos in the user's question (e.g., 'separetion' -> 'separation').
        2. Use ONLY the provided FRA manual excerpts to answer.
        3. ALWAYS cite the Source File and Page Number for every claim you make.
        4. If the answer isn't in the excerpts, state that clearly.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        
        EXCERPTS:
        {context_text}
        
        USER QUESTION: {question}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        
        outputs = self.pipe(prompt, do_sample=False, temperature=0)
        # Extract only the assistant's response
        full_text = outputs[0]["generated_text"]
        return full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
