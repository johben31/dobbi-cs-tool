import os
import requests
import chromadb
from dotenv import load_dotenv

load_dotenv()

HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_embedding(text: str) -> list[float]:
    """Get embedding from HuggingFace API"""
    response = requests.post(
        HF_API_URL,
        headers={"Authorization": f"Bearer {os.getenv('HF_API_TOKEN', '')}"},
        json={"inputs": {"source_sentence": text, "sentences": [text]}}
    )
    if response.status_code != 200:
        raise Exception(f"HuggingFace API error: {response.text}")
    result = response.json()
    return result

class DobbiRetriever:
    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection("dobbi_kb")
    
    def retrieve(self, query: str, k: int = 15) -> list[dict]:
        """Retrieve relevant documents for a query"""
        query_embedding = get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        documents = []
        for i in range(len(results['ids'][0])):
            documents.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if results['distances'] else None
            })
        
        return documents

if __name__ == "__main__":
    retriever = DobbiRetriever()
    results = retriever.retrieve("How much does a winter jacket cost?")
    print("Test query: 'How much does a winter jacket cost?'")
    print("-" * 50)
    for doc in results[:3]:
        print(f"- {doc['content'][:80]}...")
        print(f"  Distance: {doc['distance']:.3f}")
