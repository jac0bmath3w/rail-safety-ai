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
        # Build the RAG Prompt
        context_text = "\n\n".join(context_chunks)
        
        prompt = f"""
        You are a BNSF Staff Safety Engineer. Use the provided FRA manual excerpts to answer the question.
        
        EXCERPTS:
        {context_text}
        
        QUESTION:
        {question}
        
        ANSWER:
        """
        
        outputs = self.pipe(prompt, do_sample=False)
        return outputs[0]["generated_text"].split("ANSWER:")[-1].strip()