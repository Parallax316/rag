#!/usr/bin/env python3
"""
Enhanced script to check all embedding types and dimensions in MongoDB
"""

from app.core.mongodb import embeddings_col
import numpy as np

def check_all_embeddings():
    """Check all embeddings in the database grouped by type"""
    try:
        print("=== Checking All Embedding Types and Dimensions ===")
        
        # Get all documents
        all_docs = list(embeddings_col.find({}, {"type": 1, "collection_name": 1, "embedding": 1, "data": 1}))
        
        if not all_docs:
            print("No documents found in database")
            return
            
        print(f"Total documents found: {len(all_docs)}")
        
        # Group by type
        type_stats = {}
        
        for doc in all_docs:
            doc_type = doc.get("type", "unknown")
            collection_name = doc.get("collection_name", "unknown")
            embedding = doc.get("embedding", [])
            
            # Calculate embedding dimension
            if isinstance(embedding, list):
                if len(embedding) > 0 and isinstance(embedding[0], list):
                    # Multi-dimensional embedding (like ColQwen2 produces)
                    total_dim = len(embedding) * len(embedding[0]) if embedding[0] else 0
                    shape = f"({len(embedding)}, {len(embedding[0]) if embedding else 0})"
                else:
                    # Flat embedding
                    total_dim = len(embedding)
                    shape = f"({total_dim},)"
            else:
                total_dim = 0
                shape = "unknown"
            
            # Group stats
            key = f"{doc_type}_{collection_name}"
            if key not in type_stats:
                type_stats[key] = {
                    "type": doc_type,
                    "collection": collection_name,
                    "dimensions": [],
                    "shapes": [],
                    "count": 0,
                    "sample_data": None
                }
            
            type_stats[key]["dimensions"].append(total_dim)
            type_stats[key]["shapes"].append(shape)
            type_stats[key]["count"] += 1
            
            # Store sample data info
            if type_stats[key]["sample_data"] is None:
                data_info = doc.get("data", {})
                if doc_type == "image":
                    type_stats[key]["sample_data"] = {
                        "has_image_base64": "image_base64" in data_info,
                        "has_image_hash": "image_hash" in data_info
                    }
                elif doc_type == "text":
                    type_stats[key]["sample_data"] = {
                        "has_content": "content" in data_info,
                        "has_source": "source" in data_info,
                        "keys": list(data_info.keys())
                    }
                else:
                    type_stats[key]["sample_data"] = {
                        "keys": list(data_info.keys())
                    }
        
        # Print stats
        print("\n=== Embedding Statistics by Type and Collection ===")
        for key, stats in type_stats.items():
            print(f"\nType: {stats['type']}, Collection: {stats['collection']}")
            print(f"  Count: {stats['count']}")
            print(f"  Dimensions: {set(stats['dimensions'])} (unique values)")
            print(f"  Shapes: {set(stats['shapes'])} (unique shapes)")
            print(f"  Sample data structure: {stats['sample_data']}")
            
            # Check for dimension consistency within this type/collection
            unique_dims = set(stats['dimensions'])
            if len(unique_dims) > 1:
                print(f"  ⚠️  INCONSISTENT DIMENSIONS within {stats['type']}/{stats['collection']}: {unique_dims}")
            else:
                print(f"  ✅ Consistent dimensions: {list(unique_dims)[0]}")
        
        # Check for cross-type compatibility
        print("\n=== Cross-Type Compatibility Analysis ===")
        all_dims = []
        all_types = []
        for stats in type_stats.values():
            for dim in stats['dimensions']:
                all_dims.append(dim)
                all_types.append(f"{stats['type']}/{stats['collection']}")
        
        unique_all_dims = set(all_dims)
        if len(unique_all_dims) > 1:
            print(f"⚠️  INCOMPATIBLE DIMENSIONS across all types: {unique_all_dims}")
            print("This will cause vector similarity search to fail!")
            
            # Show which types have which dimensions
            dim_to_types = {}
            for i, dim in enumerate(all_dims):
                if dim not in dim_to_types:
                    dim_to_types[dim] = set()
                dim_to_types[dim].add(all_types[i])
            
            for dim, types in dim_to_types.items():
                print(f"  Dimension {dim}: {', '.join(types)}")
        else:
            print(f"✅ All embeddings have compatible dimensions: {list(unique_all_dims)[0]}")
            
    except Exception as e:
        print(f"Error checking embeddings: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_all_embeddings()
