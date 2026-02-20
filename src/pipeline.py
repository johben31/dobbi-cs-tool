import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier import QuestionClassifier
from retriever import DobbiRetriever
from generator import ResponseGenerator

class CustomerSupportPipeline:
    def __init__(self):
        print("Loading pipeline components...")
        self.classifier = QuestionClassifier()
        self.retriever = DobbiRetriever(db_path="./chroma_db")
        self.generator = ResponseGenerator()
        print("Pipeline ready!")
    
    def process(self, message: str) -> dict:
        classification = self.classifier.classify(message)
        retrieved_docs = self.retriever.retrieve(message, k=5)
        result = self.generator.generate(
            customer_message=message,
            category=classification['category'],
            retrieved_docs=retrieved_docs
        )
        
        return {
            "category": classification['category'],
            "confidence": classification['confidence'],
            "sentiment": classification['sentiment'],
            "entities": classification.get('entities', {}),
            "draft_response": result['draft_response'],
            "sources": result['sources_used'],
            "retrieval_confidence": result['confidence']
        }


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    pipeline = CustomerSupportPipeline()
    
    test_message = "Hoi, hoeveel kost het om een winterjas en 2 overhemden te laten reinigen? En hoe lang duurt het?"
    
    print(f"\nCustomer message: {test_message}\n")
    result = pipeline.process(test_message)
    
    print(f"Category: {result['category']} ({result['confidence']:.0%})")
    print(f"Sentiment: {result['sentiment']}")
    print(f"\nDraft response:")
    print("-" * 50)
    print(result['draft_response'])
    print("-" * 50)
    print(f"Sources: {result['sources']}")