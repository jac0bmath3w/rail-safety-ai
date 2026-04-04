from sentence_transformers import SentenceTransformer
import torch

class RailEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Check if GPU is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing Embedder on: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)

    def generate_embeddings(self, text_chunks):
        """Turns a list of text strings into a list of vector embeddings."""
        print(f"Generating embeddings for {len(text_chunks)} chunks...")
        # convert_to_tensor=True keeps the data on the GPU for speed
        embeddings = self.model.encode(text_chunks, show_progress_bar=True, convert_to_tensor=False)
        return embeddings
