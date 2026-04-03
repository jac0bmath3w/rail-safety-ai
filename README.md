# rail-safety-ai
Breaking down LLM Specifically for Rail Applications

8-Week Staff-Level LLM Project Roadmap

Goal: Build a "Rail-Safety Optimizer" RAG system that bridges the gap between raw federal regulations and field-level maintenance queries.

Phase 1: Foundation & Architecture (Weeks 1-2)

Goal: Move from "using an API" to understanding the math and structure of LLMs.

Week 1: Transformers & Embeddings

Day 01: Repo & Environment Setup

Initialize GitHub Repo with MIT License and Python .gitignore.

Create structure: /data/raw/, /notebooks/, /src/.

Upload the 5 FRA Safety PDFs to data/raw/.

Create a requirements.txt with core libraries (transformers, torch, langchain).

Day 02: High-Level Architecture. Read the "Attention is All You Need" paper (focus on the "Encoder/Decoder" diagram).

Day 03: Self-Attention Deep Dive. Study the math: $Attention(Q, K, V)$. Write a short explanation of why "Weighting" matters.

Day 04: Tokenizer Study. Experiment with Hugging Face's AutoTokenizer. See how specialized terms like "Trespass Prevention" are tokenized.

Day 05: Embeddings. Use sentence-transformers to generate vectors for 10 sentences from the Hazardous Material Manual. Visualize similarity.

Week 2: Vector Databases & Chunking

Day 06: Chunking Strategy. Write a Python script to split your PDFs into 500-token chunks. Explain "Overlap" (10-15%) for context preservation.

Day 07: Vector DB Setup. Set up a local Qdrant or ChromaDB instance (or use Pinecone Serverless).

Day 08: Indexing. Upload your 5 PDFs into the Vector DB. This is your "Long-term memory."

Day 09: The Retrieval Loop. Write a query script. If you ask "How to handle a Hazmat leak?" does it return the right page?

Day 10: Metadata Filtering. Learn how to tag chunks by PDF source (e.g., source: "Hazmat Manual 2025") to filter results.

Phase 2: RAG & Advanced Prompting (Weeks 3-4)

Goal: Build the reasoning engine and handle BNSF-level safety edge cases.

Week 3: Prompt Engineering & CoT

Day 11: System Prompting. Create a "Safety Officer" persona. Force the model to cite the specific Chapter/Section.

Day 12: Chain-of-Thought (CoT). Implement a "Reasoning Step." The model must first identify the hazard, then the regulation, then the solution.

Day 13: Few-Shot Prompting. Provide the model with 3 examples of "Perfect Answers" to improve its output consistency.

Day 14: Self-Correction. Add a prompt step where the model checks its own answer against the retrieved text for "hallucinations."

Day 15: Blog Post #1. Write about "Building the Knowledge Base." Focus on why safety data requires high-precision retrieval.

Week 4: Evaluation (Staff Level Requirement)

Day 16: Introduction to RAGAS. Install the library. Understand "Faithfulness" (Is the answer in the text?) and "Relevance."

Day 17: Synthetic Dataset. Generate 20 questions based on your 5 PDFs to use as an evaluation benchmark.

Day 18: Running Evaluations. Run your first "Eval." Document your score (e.g., 0.75).

Day 19: Optimization Iteration. Change a chunking parameter or prompt. Re-run the eval. Did the score improve?

Day 20: Documentation. Add an /evals folder to your repo with these results. This proves data-driven DS skill.

Phase 3: Fine-Tuning & GPU Optimization (Weeks 5-6)

Goal: The "Hard Tech" skills. Running models efficiently on hardware.

Week 5: GPU Optimization

Day 21: Quantization Basics. Use bitsandbytes to load a Llama-3 model in 4-bit on Google Colab.

Day 22: VRAM Profiling. Use nvidia-smi or Python scripts to measure memory usage before and after quantization.

Day 23: Flash Attention 2. Research and enable Flash Attention to speed up inference for long safety documents.

Day 24: KV-Caching. Learn how the model "remembers" previous words in a conversation to speed up the chat experience.

Day 25: Blog Post #2. Write about "Efficiency at Scale." Explain how BNSF could save money by using quantized models.

Week 6: Fine-Tuning (LoRA/QLoRA)

Day 26: Preparing Fine-Tuning Data. Format 50 pairs of Question/Answers from your manuals into a JSONL file.

Day 27: QLoRA Setup. Set up a training script using the peft library.

Day 28: The Training Run. Start a fine-tuning job on a small model (e.g., Llama 3.2 1B or 3B) in Colab.

Day 29: Validation. Compare the "Base Model" vs. "Fine-tuned Model." Does the fine-tuned one sound more like a BNSF engineer?

Day 30: Model Merging. Learn how to merge your LoRA weights back into the main model.

Phase 4: Deployment & Final Polish (Weeks 7-8)

Goal: Show you can build a production-ready interface.

Week 7: The Prototype

Day 31-33: Build a simple UI using Streamlit. Allow users to "Upload" a manual and ask questions.

Day 34-35: Implement Tracing. Use LangSmith or Phoenix to "see" inside the RAG loop during a live chat.

Week 8: The Staff Level Portfolio

Day 36-38: Final Blog Post summarizing the entire journey. Include your RAGAS scores and GPU benchmarks.

Day 39: Cleanup GitHub Repo. Ensure the requirements.txt and README are perfect.

Day 40: Re-apply. Update your resume with these specific keywords: RAG Evaluation, QLoRA Fine-tuning, 4-bit Quantization, Vector DB Architecture.

Note: Spend 1.5 hours per day. If a concept is hard, pause the clock and stay on that "Day" until it clicks.
