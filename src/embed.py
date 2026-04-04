from sentence_transformers import SentenceTransformer
import torch

class RailEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Check if GPU is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing Embedder on: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)

    def generate_embeddings(self, text_chunks, batch_size=32):
        all_embeddings = []
        # Process in small bites (batches) so the GPU memory doesn't overflow
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i : i + batch_size]
            batch_vecs = self.model.encode(batch)
            all_embeddings.append(batch_vecs)
        return np.vstack(all_embeddings)
