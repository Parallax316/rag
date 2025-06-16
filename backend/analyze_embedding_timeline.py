#!/usr/bin/env python3
"""
Check when embeddings were created and if there are patterns in dimensions
"""

from app.core.mongodb import embeddings_col
import numpy as np
from datetime import datetime
from bson import ObjectId

def analyze_embedding_timeline():
    """Analyze when embeddings were created and their dimensions"""
    try:
        print("=== Analyzing Embedding Timeline ===")
        
        # Get all documents from testing collection with creation timestamps
        query = {"collection_name": "testing", "type": "image"}
        docs = list(embeddings_col.find(query, {"embedding": 1, "_id": 1}).sort("_id", 1))
        
        print(f"Found {len(docs)} image documents in 'testing' collection")
        
        if not docs:
            print("No documents found")
            return
        
        # Analyze by creation time (ObjectId contains timestamp)
        timeline = []
        for doc in docs:
            obj_id = doc["_id"]
            creation_time = obj_id.generation_time
            embedding = np.array(doc["embedding"])
            shape = embedding.shape
            
            timeline.append({
                "id": str(obj_id),
                "created": creation_time,
                "shape": shape,
                "dimensions": len(embedding.flatten())
            })
        
        # Sort by creation time
        timeline.sort(key=lambda x: x["created"])
        
        print(f"\n=== Embedding Creation Timeline ===")
        current_shape = None
        shape_changes = []
        
        for i, entry in enumerate(timeline):
            created_str = entry["created"].strftime("%Y-%m-%d %H:%M:%S")
            print(f"{i+1:2d}. {created_str} | Shape: {entry['shape']} | Dims: {entry['dimensions']} | ID: {entry['id'][:12]}...")
            
            if current_shape is None:
                current_shape = entry["shape"]
                print(f"    --> First embedding shape: {current_shape}")
            elif current_shape != entry["shape"]:
                shape_changes.append({
                    "from": current_shape,
                    "to": entry["shape"],
                    "at_index": i,
                    "time": entry["created"]
                })
                current_shape = entry["shape"]
                print(f"    --> ⚠️  SHAPE CHANGE: {shape_changes[-1]['from']} → {shape_changes[-1]['to']}")
        
        print(f"\n=== Summary ===")
        if shape_changes:
            print(f"Found {len(shape_changes)} shape changes:")
            for i, change in enumerate(shape_changes):
                change_time = change["time"].strftime("%Y-%m-%d %H:%M:%S")
                print(f"  {i+1}. At {change_time}: {change['from']} → {change['to']}")
            
            print("\n⚠️  This suggests the model configuration was changed during indexing!")
            print("Recommendation: Clear the collection and re-index all images with current model")
        else:
            print("✅ All embeddings have consistent shapes")
            print("The dimension mismatch is between stored embeddings and current model queries")
        
        # Show shape distribution
        shape_counts = {}
        for entry in timeline:
            shape_str = str(entry["shape"])
            if shape_str not in shape_counts:
                shape_counts[shape_str] = 0
            shape_counts[shape_str] += 1
        
        print(f"\n=== Shape Distribution ===")
        for shape_str, count in shape_counts.items():
            print(f"Shape {shape_str}: {count} documents")
            
    except Exception as e:
        print(f"Error analyzing timeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_embedding_timeline()
