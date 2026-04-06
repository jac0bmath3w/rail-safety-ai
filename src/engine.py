from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
import os

class RailSafetyEngine:
    def __init__(self, model_id="unsloth/Llama-3.2-3B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa"
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.1, 
            device_map="auto"
        )

    def generate_answer(self, question, context_chunks):
        if not context_chunks:
            return "Error: No context provided."

        # STAFF LEVEL REFINEMENT: Intent-Aware Filtering
        # We classify the query intent to handle 'Removal' vs 'Requirement' logic dynamically
        is_removal_query = any(word in question.lower() for word in ["remove", "removal", "closing", "abandon"])
        is_requirement_query = any(word in question.lower() for word in ["require", "consider", "justify", "criteria", "threshold"])
        
        filtered_chunks = []
        for chunk in context_chunks:
            # If asking about establishing rules, skip 'Removal' sections to avoid confusion
            if is_requirement_query and not is_removal_query:
                if "removal of" in chunk.lower() or "abandoned rail" in chunk.lower():
                    continue
            # If asking about closing, prioritize the opposite
            if is_removal_query and not is_requirement_query:
                if "justification for new" in chunk.lower() or "establishing grade" in chunk.lower():
                    continue
            filtered_chunks.append(chunk)
            
        final_context = filtered_chunks if filtered_chunks else context_chunks
        context_text = "\n\n".join(final_context)
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a Senior FRA Safety Consultant. 
        
        STRICT OPERATIONAL PROTOCOL:
        1. QUERY ANALYSIS: Correct typos and identify the primary intent (e.g., maintenance, regulatory compliance, engineering thresholds).
        2. DATA SYNTHESIS: Extract specific numerical data, thresholds, and technical requirements. 
        3. CONTEXTUAL RELEVANCE: Ensure the answer matches the user's intent. Do not confuse 'Removing' a structure with 'Establishing' one unless the text explicitly links them.
        4. AUTHORSHIP & PROVENANCE: Identify specific authors, offices, or consulting firms (e.g., Kimley-Horn) mentioned in the excerpts.
        5. VERACITY: If the answer is not in the excerpts, state: "I cannot find this specific information in the provided manual excerpts."
        6. CITATION: Every technical claim MUST be followed by [File Name, Page Number].
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        
        CONTEXT EXCERPTS:
        {context_text}
        
        USER QUESTION: {question}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        outputs = self.pipe(
            prompt, 
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = outputs[0]["generated_text"]
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            return response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        return response
