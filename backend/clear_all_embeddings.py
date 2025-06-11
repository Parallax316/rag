#!/usr/bin/env python3
"""
Clear all embeddings from MongoDB and prepare for re-indexing with current model
"""

from app.core.mongodb import embeddings_col
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_all_embeddings():
    """Clear all embeddings from the database"""
    try:
        print("=== Clearing All Embeddings ===")
        
        # Count current documents
        current_count = embeddings_col.count_documents({})
        print(f"Current documents in database: {current_count}")
        
        if current_count == 0:
            print("Database is already empty.")
            return
        
        # Ask for confirmation
        response = input(f"Are you sure you want to delete all {current_count} embeddings? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled.")
            return
        
        # Delete all documents
        print("Deleting all embeddings...")
        result = embeddings_col.delete_many({})
        
        print(f"✅ Successfully deleted {result.deleted_count} documents")
        
        # Verify deletion
        remaining_count = embeddings_col.count_documents({})
        print(f"Remaining documents: {remaining_count}")
        
        if remaining_count == 0:
            print("✅ Database is now clean and ready for re-indexing with current model")
        else:
            print(f"⚠️  Warning: {remaining_count} documents still remain")
            
    except Exception as e:
        print(f"❌ Error clearing embeddings: {e}")
        logger.error(f"Error clearing embeddings: {e}", exc_info=True)

if __name__ == "__main__":
    clear_all_embeddings()
