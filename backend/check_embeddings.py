#!/usr/bin/env python3
"""
Script to check embedding dimensions in MongoDB and clean up inconsistent data
"""

from app.core.mongodb import embeddings_col
import os
from dotenv import load_dotenv

def check_embedding_dimensions():
    """Check what embedding dimensions exist in the database"""
    
    # Load environment variables
    load_dotenv()
    
    print("=== Checking Embedding Dimensions ===")
    
    # Check unique embedding dimensions
    pipeline = [
        {"$group": {
            "_id": {"$size": "$embedding"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"count": -1}}
    ]
    
    print('Embedding dimensions in database:')
    dimensions = list(embeddings_col.aggregate(pipeline))
    for result in dimensions:
        print(f'Dimension: {result["_id"]}, Count: {result["count"]}')
    
    # Get some sample documents to understand the structure
    print('\nSample documents:')
    for doc in embeddings_col.find().limit(5):
        doc_type = doc.get("type", "unknown")
        embedding_dim = len(doc.get("embedding", []))
        collection_name = doc.get("collection_name", "unknown")
        print(f'Type: {doc_type}, Collection: {collection_name}, Dimension: {embedding_dim}')
    
    return dimensions

def clean_inconsistent_embeddings(target_dimension=None):
    """Remove embeddings that don't match the target dimension"""
    
    if target_dimension is None:
        print("Please specify target_dimension parameter")
        return
    
    print(f"\n=== Cleaning Embeddings (keeping dimension {target_dimension}) ===")
    
    # Find documents with wrong dimensions
    pipeline = [
        {"$match": {
            "$expr": {"$ne": [{"$size": "$embedding"}, target_dimension]}
        }}
    ]
    
    wrong_dim_docs = list(embeddings_col.aggregate(pipeline))
    print(f"Found {len(wrong_dim_docs)} documents with incorrect dimensions")
    
    if wrong_dim_docs:
        # Delete documents with wrong dimensions
        delete_result = embeddings_col.delete_many({
            "$expr": {"$ne": [{"$size": "$embedding"}, target_dimension]}
        })
        print(f"Deleted {delete_result.deleted_count} documents with incorrect dimensions")
    
    # Verify cleanup
    remaining_pipeline = [
        {"$group": {
            "_id": {"$size": "$embedding"},
            "count": {"$sum": 1}
        }}
    ]
    
    print("Remaining dimensions after cleanup:")
    for result in embeddings_col.aggregate(remaining_pipeline):
        print(f'Dimension: {result["_id"]}, Count: {result["count"]}')

if __name__ == "__main__":
    dimensions = check_embedding_dimensions()
    
    if len(dimensions) > 1:
        print(f"\n=== DIMENSION MISMATCH DETECTED ===")
        print("Multiple embedding dimensions found. This will cause VLM query failures.")
        print("Available options:")
        print("1. Keep largest dimension group (likely the current model)")
        print("2. Keep smallest dimension group")
        print("3. Delete all and re-embed")
        
        # Find the most common dimension (likely the correct one)
        most_common = max(dimensions, key=lambda x: x["count"])
        print(f"\nMost common dimension: {most_common['_id']} ({most_common['count']} documents)")
        
        response = input(f"Clean database keeping dimension {most_common['_id']}? (y/n): ")
        if response.lower() == 'y':
            clean_inconsistent_embeddings(most_common['_id'])
        else:
            print("No cleanup performed. VLM queries will continue to fail until this is resolved.")
    else:
        print("\n=== All embeddings have consistent dimensions ===")
        print("Database is clean and ready for VLM queries.")
