# MongoDB Atlas connection utility for multimodal RAG system

import os
import time
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import Optional
import logging

# Load environment variables from .env at project root
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

MONGODB_URI = os.getenv('MONGODB_ATLAS_URI')
MONGODB_DB = os.getenv('MONGODB_ATLAS_DB', 'rag_multimodal')
EMBEDDINGS_COLLECTION = os.getenv('MONGODB_EMBEDDINGS_COLLECTION', 'embeddings')

logger = logging.getLogger("mongodb")

try:
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DB]
    embeddings_col = db[EMBEDDINGS_COLLECTION]
    logger.info(f"Connected to MongoDB: {MONGODB_URI}, DB: {MONGODB_DB}, Collection: {EMBEDDINGS_COLLECTION}")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise

# System config collection for storing current selected collection, etc.
try:
    system_config_col = db[os.getenv('MONGODB_SYSTEM_CONFIG_COLLECTION', 'system_config')]
except Exception as e:
    logger.error(f"Failed to get system config collection: {e}")
    system_config_col = None

def insert_embedding(document: dict):
    try:
        logger.info("=== Starting MongoDB insertion ===")
        print(f"[MONGODB] Starting document insertion...")
        start_time = time.time()
        
        # Log document size for debugging
        doc_size = len(str(document))
        logger.info(f"Document size: {doc_size} characters")
        print(f"[MONGODB] Document size: {doc_size} characters")
        
        # Check if embedding field exists and log its size
        if 'embedding' in document:
            embedding_size = len(document['embedding']) if document['embedding'] else 0
            logger.info(f"Embedding dimensions: {embedding_size}")
            print(f"[MONGODB] Embedding dimensions: {embedding_size}")
        
        logger.info("Inserting document into MongoDB...")
        print(f"[MONGODB] Inserting document into MongoDB...")
        
        result = embeddings_col.insert_one(document)
        
        insert_time = time.time() - start_time
        logger.info(f"Document inserted with _id: {result.inserted_id} in {insert_time:.2f} seconds")
        print(f"[MONGODB] Document inserted successfully in {insert_time:.2f} seconds")
        logger.info("=== MongoDB insertion completed ===")
        
        return result
    except Exception as e:
        logger.error(f"Error inserting embedding: {e}", exc_info=True)
        print(f"[MONGODB] ERROR: {str(e)}")
        raise

def find_embeddings(query: dict):
    try:
        logger.info(f"=== Starting MongoDB query ===")
        print(f"[MONGODB] Starting query: {query}")
        start_time = time.time()
        
        results = list(embeddings_col.find(query))
        
        query_time = time.time() - start_time
        logger.info(f"Found {len(results)} embeddings for query: {query} in {query_time:.2f} seconds")
        print(f"[MONGODB] Found {len(results)} documents in {query_time:.2f} seconds")
        logger.info("=== MongoDB query completed ===")
        
        return results
    except Exception as e:
        logger.error(f"Error finding embeddings: {e}", exc_info=True)
        print(f"[MONGODB] QUERY ERROR: {str(e)}")
        raise

def update_embedding(query: dict, update: dict):
    try:
        result = embeddings_col.update_one(query, {'$set': update})
        logger.info(f"Updated {result.modified_count} embedding(s) for query: {query}")
        return result
    except Exception as e:
        logger.error(f"Error updating embedding: {e}")
        raise

def delete_embedding(query: dict):
    try:
        result = embeddings_col.delete_one(query)
        logger.info(f"Deleted {result.deleted_count} embedding(s) for query: {query}")
        return result
    except Exception as e:
        logger.error(f"Error deleting embedding: {e}")
        raise

def list_collections():
    try:
        collections = db.list_collection_names()
        logger.info(f"Collections in DB: {collections}")
        return collections
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise

def create_collection(name: str):
    try:
        db.create_collection(name)
        logger.info(f"Created collection: {name}")
    except Exception as e:
        logger.error(f"Error creating collection {name}: {e}")
        raise

def drop_collection(name: str):
    try:
        db.drop_collection(name)
        logger.info(f"Dropped collection: {name}")
    except Exception as e:
        logger.error(f"Error dropping collection {name}: {e}")
        raise

def set_system_config(key: str, value: str):
    if not system_config_col:
        logger.error("System config collection not initialized!")
        return
    try:
        system_config_col.update_one({"key": key}, {"$set": {"value": value}}, upsert=True)
        logger.info(f"Set system config: {key} = {value}")
    except Exception as e:
        logger.error(f"Error setting system config: {e}")
        raise

def get_system_config(key: str) -> Optional[str]:
    if not system_config_col:
        logger.error("System config collection not initialized!")
        return None
    try:
        doc = system_config_col.find_one({"key": key})
        logger.info(f"Get system config: {key} -> {doc['value'] if doc else None}")
        return doc["value"] if doc else None
    except Exception as e:
        logger.error(f"Error getting system config: {e}")
        return None
