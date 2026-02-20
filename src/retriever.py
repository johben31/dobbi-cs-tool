from sentence_transformers import SentenceTransformer
import chromadb

class DobbiRetriever:
    def __init__(self, db_path="./chroma_db"):
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_collection("dobbi_kb")
    
    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        retrieved_docs = []
        for i in range(len(results['documents'][0])):
            retrieved_docs.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if results['distances'] else None
            })
        
        return retrieved_docs


if __name__ == "__main__":
    retriever = DobbiRetriever()
    
    print("=" * 50)
    print("Test 1: Dutch price question")
    print("=" * 50)
    results = retriever.retrieve("Hoeveel kost een winterjas reinigen?")
    for r in results[:3]:
        print(f"- {r['content']}")
        print(f"  Distance: {r['distance']:.3f}")
        print()
    
    print("=" * 50)
    print("Test 2: English FAQ question")
    print("=" * 50)
    results = retriever.retrieve("What is the minimum order?")
    for r in results[:3]:
        print(f"- {r['content']}")
        print(f"  Distance: {r['distance']:.3f}")
        print()