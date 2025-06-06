import sqlite3
import pickle
import logging
from ..config import DATABASE_PATH

logger = logging.getLogger(__name__)

def get_db_connection():
    """
    Create and return a database connection
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        # Create tables if they don't exist
        create_tables(conn)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def create_tables(conn):
    """
    Create necessary tables if they don't exist
    """
    try:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_base64 TEXT,
                image_hash TEXT UNIQUE,
                embedding BLOB
            )
        ''')
        conn.commit()
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        conn.rollback()
        raise

def store_embedding(conn, img_str, image_hash, embedding):
    """
    Store image embedding in the database
    """
    try:
        c = conn.cursor()
        # Check if the image hash already exists
        c.execute('SELECT id FROM embeddings WHERE image_hash = ?', (image_hash,))
        result = c.fetchone()
        if result:
            # Image already indexed
            logger.info(f"Image {image_hash[:8]} already indexed, skipping")
            return False
        
        # Serialize the embedding
        embedding_bytes = pickle.dumps(embedding)
        c.execute('INSERT INTO embeddings (image_base64, image_hash, embedding) VALUES (?, ?, ?)', 
                 (img_str, image_hash, embedding_bytes))
        conn.commit()
        logger.info(f"Image {image_hash[:8]} indexed and stored in database")
        return True
    except Exception as e:
        logger.error(f"Error storing embedding: {str(e)}")
        conn.rollback()
        raise

def get_all_embeddings(conn):
    """
    Retrieve all image embeddings from the database
    """
    try:
        c = conn.cursor()
        c.execute('SELECT image_base64, embedding FROM embeddings')
        rows = c.fetchall()
        
        if not rows:
            logger.warning("No images found in the index")
            return [], []
            
        logger.info(f"Retrieved {len(rows)} image embeddings from database")
        
        image_base64_list = []
        embeddings_list = []
        
        for row in rows:
            image_base64, embedding_bytes = row
            embedding = pickle.loads(embedding_bytes)
            
            image_base64_list.append(image_base64)
            embeddings_list.append(embedding)
            
        return image_base64_list, embeddings_list
    except Exception as e:
        logger.error(f"Error retrieving embeddings: {str(e)}")
        raise