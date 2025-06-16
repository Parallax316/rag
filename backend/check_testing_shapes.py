#!/usr/bin/env python3
"""
Check embedding shapes in the testing collection to diagnose the shape mismatch issue
"""

from app.core.mongodb import embeddings_col
import numpy as np

def check_embedding_shapes():
    """Check embedding shapes in the testing collection"""
    try:
        print("=== Checking Embedding Shapes in 'testing' Collection ===")
        
        # Get all documents from testing collection
        query = {"type": "image", "collection_name": "testing"}
        docs = list(embeddings_col.find(query, {"embedding": 1, "_id": 1}))
        
        print(f"Found {len(docs)} image documents in 'testing' collection")
        
        shapes = {}
        shape_counts = {}
        
        for i, doc in enumerate(docs):
            embedding = doc.get("embedding", [])
            
            if isinstance(embedding, list):
                # Convert to numpy array to get shape
                emb_array = np.array(embedding)
                shape = emb_array.shape
                shape_str = str(shape)
                
                if shape_str not in shapes:
                    shapes[shape_str] = []
                    shape_counts[shape_str] = 0
                
                shapes[shape_str].append(doc["_id"])
                shape_counts[shape_str] += 1
                
                # Show first few for debugging
                if i < 5:
                    print(f"Document {i+1}: Shape {shape}, ID: {doc['_id']}")
        
        print(f"\n=== Shape Summary ===")
        for shape_str, count in shape_counts.items():
            print(f"Shape {shape_str}: {count} documents")
            
        if len(shape_counts) > 1:
            print(f"\n⚠️  PROBLEM: Found {len(shape_counts)} different embedding shapes!")
            print("This causes the 'all input arrays must have the same shape' error.")
            print("\nSolutions:")
            print("1. Clear the collection and re-index all images")
            print("2. Or filter by shape before stacking")
            
            # Show which documents have which shapes
            for shape_str, doc_ids in shapes.items():
                print(f"\nShape {shape_str} documents (showing first 5):")
                for doc_id in doc_ids[:5]:
                    print(f"  - {doc_id}")
        else:
            print("✅ All embeddings have the same shape - no shape mismatch issue")
            
    except Exception as e:
        print(f"Error checking embeddings: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_embedding_shapes()
