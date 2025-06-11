#!/usr/bin/env python3
"""
Script to clear all embeddings from MongoDB to resolve dimension mismatches
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.mongodb import get_collection
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_all_embeddings():
    """Clear all embeddings from MongoDB"""
    try:
        print("=== Clearing All Embeddings ===")
        
        # Get the collection
        collection = get_collection()
        
        # Count current documents
        current_count = collection.count_documents({})
        print(f"Current documents in database: {current_count}")
        
        if current_count == 0:
            print("Database is already empty.")
            return
        
        # Ask for confirmation
        confirm = input(f"Are you sure you want to delete all {current_count} embeddings? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Operation cancelled.")
            return
        
        # Delete all documents
        result = collection.delete_many({})
        print(f"Successfully deleted {result.deleted_count} documents")
        
        # Verify deletion
        remaining_count = collection.count_documents({})
        print(f"Remaining documents: {remaining_count}")
        
        if remaining_count == 0:
            print("✅ Database cleared successfully!")
            print("You can now re-index your images with the current model.")
        else:
            print(f"⚠️  Warning: {remaining_count} documents still remain")
            
    except Exception as e:
        logger.error(f"Error clearing embeddings: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    clear_all_embeddings()
