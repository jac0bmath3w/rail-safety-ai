from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch

class RailSafetyEngine:
    def __init__(self, model_id="unsloth/Llama-3.2-3B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Ensure a padding token exists
        if self.tokenizer.pad_token is None:
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
            return "No relevant safety manual excerpts were found."

        # STAFF LEVEL REFINEMENT: Intent-Aware Filtering
        # We classify the query intent to handle 'Removal' vs 'Requirement' logic dynamically
        # is_removal_query = any(word in question.lower() for word in ["remove", "removal", "closing", "abandon"])
        # is_requirement_query = any(word in question.lower() for word in ["require", "consider", "justify", "criteria", "threshold"])

#        THIS LOOKS VERY SPECIFIC. WE DON'T WANT THAT
#        filtered_chunks = []
#        for chunk in context_chunks:
#            # If asking about establishing rules, skip 'Removal' sections to avoid confusion
#            if is_requirement_query and not is_removal_query:
#                if "removal of" in chunk.lower() or "abandoned rail" in chunk.lower():
#                    continue
#            # If asking about closing, prioritize the opposite
#            if is_removal_query and not is_requirement_query:
#                if "justification for new" in chunk.lower() or "establishing grade" in chunk.lower():
#                    continue
#            filtered_chunks.append(chunk)
            
#        final_context = filtered_chunks if filtered_chunks else context_chunks
        final_context = context_chunks
        context_text = "\n\n".join(final_context)
        
        # 1. Structure the data as a standard list of messages with Staff-Level Instructions
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a Senior FRA Safety Consultant. \n\n"
                    "STRICT OPERATIONAL PROTOCOL:\n"
                    "1. QUERY ANALYSIS: Correct typos and identify the primary intent (e.g., maintenance, regulatory compliance).\n"
                    "2. INTENT VALIDATION: Distinguish between 'Establishing' a project and 'Removing' a structure. Only use relevant context.\n"
                    "3. DATA SYNTHESIS: Extract specific numerical data, thresholds (e.g., AADT > 30,000), and technical requirements.\n"
                    "4. AUTHORSHIP: Identify specific authors (e.g., Brent Ogden) or consulting firms (e.g., Kimley-Horn) if found.\n"
                    "5. VERACITY: Use ONLY the provided context. If the answer is missing, state so clearly.\n"
                    "6. CITATION: Every technical claim MUST be followed by [Manual Name, Page Number]."
                )
            },
            {
                "role": "user", 
                "content": f"CONTEXT FROM MANUALS:\n{context_text}\n\nUSER QUESTION: {question}"
            }
        ]
        
        # 2. Use the model-agnostic chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 3. Generate
        outputs = self.pipe(
            prompt, 
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # 4. Extract the response (assistant output only)
        return outputs[0]["generated_text"][len(prompt):].strip()
