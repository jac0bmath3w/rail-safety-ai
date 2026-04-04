import chromadb

class RailVectorVault:
    def __init__(self, embedder_instance, db_path="./vector_db"):
        # We pass the embedder IN. This is called 'Dependency Injection'.
        self.embedder = embedder_instance 
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="rail_safety")

    def add_documents(self, chunks, metadatas):
        # The Vault asks the Embedder to do its job
        vectors = self.embedder.generate_embeddings(chunks)
        ids = [f"id_{i}" for i in range(len(chunks))]
        
        self.collection.add(
            documents=chunks,
            embeddings=vectors.tolist(),
            metadatas=metadatas,
            ids=ids
        )
