from sentence_transformers import SentenceTransformer
import chromadb
import json
import pandas as pd
import os

class KnowledgeBaseIndexer:
    def __init__(self, db_path="./chroma_db"):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="dobbi_kb",
            metadata={"hnsw:space": "cosine"}
        )
        print("Indexer ready!")
    
    def index_faq(self, faq_json_path: str):
        with open(faq_json_path, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
        
        documents = []
        ids = []
        metadatas = []
        
        for item in faq_data:
            doc_text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            documents.append(doc_text)
            ids.append(item['id'])
            metadatas.append({
                "source": "faq",
                "category": item.get('category', 'general'),
                "language": item.get('language', 'en')
            })
        
        print(f"Embedding {len(documents)} FAQ items...")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Indexed {len(documents)} FAQ items from {faq_json_path}")
    
    def index_prices(self, csv_path: str):
        df = pd.read_csv(csv_path)
        
        documents = []
        ids = []
        metadatas = []
        
        for idx, row in df.iterrows():
            doc_text = f"{row['item_name']} ({row['item_name_nl']}): €{row['price_eur']}"
            if pd.notna(row.get('notes')) and row.get('notes'):
                doc_text += f" - {row['notes']}"
            
            documents.append(doc_text)
            ids.append(f"price_{idx}")
            metadatas.append({
                "source": "price_list",
                "category": row.get('category', 'general')
            })
        
        print(f"Embedding {len(documents)} price items...")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Indexed {len(documents)} price items from {csv_path}")
    
    def clear_index(self):
        self.chroma_client.delete_collection("dobbi_kb")
        self.collection = self.chroma_client.get_or_create_collection(
            name="dobbi_kb",
            metadata={"hnsw:space": "cosine"}
        )
        print("Index cleared!")


if __name__ == "__main__":
    indexer = KnowledgeBaseIndexer()
    
    if os.path.exists("knowledge_base/faq_en.json"):
        indexer.index_faq("knowledge_base/faq_en.json")
    if os.path.exists("knowledge_base/faq_nl.json"):
        indexer.index_faq("knowledge_base/faq_nl.json")
    if os.path.exists("knowledge_base/prices.csv"):
        indexer.index_prices("knowledge_base/prices.csv")
    
    print(f"\nTotal items in index: {indexer.collection.count()}")