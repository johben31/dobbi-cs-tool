import json
import os
import requests
import chromadb
from chromadb.config import Settings
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

HF_API_URL = "https://router.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings from HuggingFace API"""
    response = requests.post(
        HF_API_URL,
        headers={"Authorization": f"Bearer {os.getenv('HF_API_TOKEN', '')}"},
        json={"inputs": texts, "options": {"wait_for_model": True}}
    )
    if response.status_code != 200:
        raise Exception(f"HuggingFace API error: {response.text}")
    return response.json()

class KnowledgeBaseIndexer:
    def __init__(self, db_path: str = "./chroma_db"):
        print("Initializing indexer with HuggingFace API...")
        self.client = chromadb.PersistentClient(path=db_path)
        
        try:
            self.client.delete_collection("dobbi_kb")
        except:
            pass
        
        self.collection = self.client.create_collection(
            name="dobbi_kb",
            metadata={"hnsw:space": "cosine"}
        )
        print("Indexer ready!")
    
    def index_faq(self, json_path: str):
        """Index FAQ items from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            faq_items = json.load(f)
        
        documents = []
        ids = []
        metadatas = []
        
        for item in faq_items:
            doc_text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            documents.append(doc_text)
            ids.append(item['id'])
            metadatas.append({
                "source": "faq",
                "category": item.get('category', 'general'),
                "language": item.get('language', 'en')
            })
        
        print(f"Embedding {len(documents)} FAQ items...")
        embeddings = get_embeddings(documents)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Indexed {len(documents)} FAQ items from {json_path}")
    
    def index_prices(self, csv_path: str):
        """Index price list from CSV"""
        df = pd.read_csv(csv_path)
        
        documents = []
        ids = []
        metadatas = []
        
        for idx, row in df.iterrows():
            doc_text = f"{row['Item']}: €{row['Price (EUR)']}"
            
            documents.append(doc_text)
            ids.append(f"price_{idx}")
            metadatas.append({
                "source": "price_list",
                "category": row['Category']
            })
        
        print(f"Embedding {len(documents)} price items...")
        embeddings = get_embeddings(documents)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Indexed {len(documents)} price items from {csv_path}")

if __name__ == "__main__":
    indexer = KnowledgeBaseIndexer()
    
    if os.path.exists("knowledge_base/faq_en.json"):
        indexer.index_faq("knowledge_base/faq_en.json")
    if os.path.exists("knowledge_base/faq_nl.json"):
        indexer.index_faq("knowledge_base/faq_nl.json")
    if os.path.exists("knowledge_base/terms_en.json"):
        indexer.index_faq("knowledge_base/terms_en.json")
    if os.path.exists("knowledge_base/prices.csv"):
        indexer.index_prices("knowledge_base/prices.csv")
    
    print(f"\nTotal items in index: {indexer.collection.count()}")
