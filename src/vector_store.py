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
        
    def query(self, question, n_results=3):
        """
        Performs a semantic search.
        1. Embeds the question using the injected embedder.
        2. Queries ChromaDB for the closest matches.
        """
        # Embed the query string
        query_vector = self.embedder.generate_embeddings([question])
        
        # Search the collection
        results = self.collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=n_results
        )
        return results
