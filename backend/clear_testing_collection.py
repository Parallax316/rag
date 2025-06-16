#!/usr/bin/env python3
"""
Clear the testing collection and prepare for re-indexing with current model
"""

from app.core.mongodb import embeddings_col
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_testing_collection():
    """Clear all embeddings from the testing collection"""
    try:
        print("=== Clearing Testing Collection ===")
        
        # Count current documents in testing collection
        query = {"collection_name": "testing"}
        current_count = embeddings_col.count_documents(query)
        print(f"Current documents in 'testing' collection: {current_count}")
        
        if current_count == 0:
            print("Testing collection is already empty.")
            return
        
        # Ask for confirmation
        response = input(f"Are you sure you want to delete all {current_count} documents from 'testing' collection? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled.")
            return
        
        # Delete all documents from testing collection
        print("Deleting all embeddings from testing collection...")
        result = embeddings_col.delete_many(query)
        
        print(f"✅ Successfully deleted {result.deleted_count} documents from testing collection")
        
        # Verify deletion
        remaining_count = embeddings_col.count_documents(query)
        print(f"Remaining documents in testing collection: {remaining_count}")
        
        if remaining_count == 0:
            print("✅ Testing collection is now clean and ready for re-indexing with current model")
            print("\nNext steps:")
            print("1. Re-upload your images to the testing collection")
            print("2. They will be indexed with the current model configuration")
            print("3. Query and image embeddings will then have compatible dimensions")
        else:
            print(f"⚠️  Warning: {remaining_count} documents still remain")
            
    except Exception as e:
        print(f"❌ Error clearing testing collection: {e}")
        logger.error(f"Error clearing testing collection: {e}", exc_info=True)

if __name__ == "__main__":
    clear_testing_collection()
