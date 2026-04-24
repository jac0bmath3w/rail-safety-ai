import torch
from typing import List, Dict, Optional, Any, Union

def run_integrated_audit(questions, vault, tokenizer, model, method = 'rerank', n_results = 10, source_filter = None, show_context = False):

    # Methods: 'vector', 'hybrid', 'rerank'

    # We pull 10 chunks to give the model plenty of "Open Book" evidence
    # results = vault.query(question, n_results=n_results)
    # context = "\n---\n".join(results['documents'][0])
    if isinstance(questions, str):
        questions = [questions]

    where_clause = {"source": source_filter} if source_filter else None

    if method == 'vector':
        batch_results = vault.query(questions, n_results=n_results, where=where_clause)
    elif method == 'hybrid':
        results = vault.hybrid_query(questions, n_results=n_results, where=where_clause)
    else: # Default to rerank
        batch_results = vault.rerank_query(questions, n_results=n_results, where=where_clause)

    all_messages = []
    for q_idx, question in enumerate(questions):
        results = batch_results[q_idx]
        context_parts = []

        if show_context:
            print(f"\n--- LIBRARIAN REPORT: Question {q_idx+1} ({len(results)} chunks via {method.upper()}) ---")

        for i, res in enumerate(results):
            text = res['text']
            meta = res['metadata']

            if show_context:
                print(f"Chunk {i+1}: Source: {meta['source']}, Page: {meta['page']} (ID: {res['id']})")
                print(f"Preview: {text[:80]}...\n")

            context_parts.append(f"[SOURCE: {meta['source']}, PAGE: {meta['page']}]\n{text}")

        context = "\n---\n".join(context_parts)

        # Build the message list for this specific question
        # messages = [
        #     {"role": "system", "content": "You are a Senior FRA Safety Consultant. Use your 4-Phase Thinking Process. Answer ONLY based on the provided context."},
        #     {"role": "user", "content": f"CONTEXT FROM MANUALS:\n{context}\n\nQUESTION:\n{question}"},
        # ]
        messages = [
                        {
                            "role": "system", 
                            "content": (
                                "You are a Senior FRA Safety Consultant. Use your 4-Phase Thinking Process. "
                                "Structure your response exactly as follows:\n"
                                "[THINKING PROCESS]\n"
                                "<detailed reasoning>\n\n"
                                "[ANSWER]\n"
                                "<final response with inline citations>\n\n"
                                "Answer ONLY based on the provided context. In the [ANSWER] section, you MUST "
                                "cite the source and page number for every claim (e.g., [SOURCE: X, PAGE: Y])."
                            )
                        },
                        {"role": "user", "content": f"CONTEXT FROM MANUALS:\n{context}\n\nQUESTION:\n{question}"},
                    ]
        all_messages.append(messages)

    # # Extract text and metadata for the prompt
    # context_parts = []
    # for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
    #     context_parts.append(f"[SOURCE: {meta['source']}, PAGE: {meta['page']}]\n{doc}")
    prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in all_messages]

    # Ensure pad_token is set (crucial for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to("cuda")
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            use_cache=True,
            temperature=0,
            do_sample=False
        )
    generated_tokens = outputs[:, input_len:]
    response = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return response#[0].split("assistant")[-1].strip()
