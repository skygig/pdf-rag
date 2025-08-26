# database_loader.py : It connects to the PostgreSQL database and loads document and chunk data with embeddings.

import os
import hashlib
import psycopg2
from psycopg2 import extras
from dotenv import load_dotenv
from typing import List
from .textChunker import TextChunk 
from .LocalEmbeddingGenerator import LocalEmbeddingGenerator

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    load_dotenv()
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"🔴 Could not connect to the database: {e}")
        return None


def calculate_chunk_hash(chunk_content: str, chunk_index: int, doc_version_id: int) -> str:
    """Calculate a unique hash for a chunk based on its content, index, and document version."""
    hash_input = f"{doc_version_id}:{chunk_index}:{chunk_content}"
    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()


def get_file_info(file_path: str) -> tuple:
    """Get file size and calculate SHA256 hash of a file."""
    try:
        file_size = os.path.getsize(file_path)
        
        # Calculate SHA256 hash
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return file_size, sha256_hash.hexdigest()
    except (OSError, IOError) as e:
        print(f"🔴 Error reading file {file_path}: {e}")
        return None, None

# --- Data Insertion Logic ---

def load_document_to_db(
    conn,
    source_uri: str,
    source_hash: str,
    chunks: List[TextChunk],
    file_size: int = None,
    page_count: int = None,
    embedding_model: str = "multi-qa-MiniLM-L6-cos-v1",
    batch_size: int = 32
) -> bool:
    """
    Loads a document and its text chunks into the database with embedding generation.

    This function performs the following steps:
    1. Checks if the document (by its hash) already exists.
    2. If not, it inserts a new record into the `documents` table.
    3. It creates a new record in the `document_versions` table.
    4. It bulk-inserts all text chunks (with embeddings) into the `chunks` table.
    5. All operations are performed within a single transaction.

    Args:
        conn: An active psycopg2 database connection.
        source_uri: The URI or file path of the source document.
        source_hash: The SHA256 hash of the source document.
        chunks: A list of TextChunk objects to be inserted.
        file_size: Size of the file in bytes.
        page_count: Number of pages in the document.
        embedding_model: Name of the embedding model to use.
        batch_size: Batch size for embedding generation.

    Returns:
        True if the loading was successful, False otherwise.
    """
    if not chunks:
        print("⚠️ No chunks provided. Nothing to insert.")
        return False

    # Initialize embedding generator if needed
    embedding_generator = None
    try:
        print(f"🔄 Initializing embedding generator with model: {embedding_model}")
        embedding_generator = LocalEmbeddingGenerator(embedding_model)
        print(f"✅ Embedding generator ready (dimension: {embedding_generator.embedding_dim})")
    except Exception as e:
        print(f"⚠️ Failed to initialize embedding generator: {e}")

    try:
        with conn:
            with conn.cursor() as cur:
                # 1. Check for existing document
                cur.execute(
                    "SELECT id FROM documents WHERE source_sha256 = %s;",
                    (source_hash,)
                )
                doc_id_result = cur.fetchone()

                if doc_id_result:
                    doc_id = doc_id_result[0]
                    print(f"ℹ️ Document with hash {source_hash[:10]}... already exists (ID {doc_id}).")
                    
                    # Check if we need to create a new version
                    cur.execute(
                        "SELECT COUNT(*) FROM document_versions WHERE document_id = %s;",
                        (doc_id,)
                    )
                    version_count = cur.fetchone()[0]
                    
                    # For simplicity, we'll skip if document exists
                    # In a real system, you might want to create new versions
                    print(f"ℹ️ Skipping insertion as document already has {version_count} version(s).")
                    return True
                else:
                    # Insert new document
                    cur.execute(
                        """
                        INSERT INTO documents (title, source_path, source_sha256)
                        VALUES (%s, %s, %s) RETURNING id;
                        """,
                        (os.path.basename(source_uri), source_uri, source_hash)
                    )
                    doc_id = cur.fetchone()[0]
                    print(f"✅ Inserted new document '{os.path.basename(source_uri)}' with ID {doc_id}.")

                # 2. Insert document version
                # Calculate page_count from chunks if not provided
                if page_count is None:
                    page_count = max(chunk.page_end for chunk in chunks) if chunks else 0
                
                cur.execute(
                    """
                    INSERT INTO document_versions (document_id, file_size, page_count, processing_status)
                    VALUES (%s, %s, %s, %s) RETURNING id;
                    """,
                    (doc_id, file_size or 0, page_count, 'processing')
                )
                doc_version_id = cur.fetchone()[0]
                print(f"✅ Created document version ID {doc_version_id}.")

                # 3. Generate embeddings
                embeddings = []
                if embedding_generator:
                    print(f"🔄 Generating embeddings for {len(chunks)} chunks...")
                    
                    # Prepare texts for embedding generation
                    texts_to_embed = []
                    for chunk in chunks:
                        # Use text_clean if available, otherwise use text
                        content_to_embed = getattr(chunk, 'text_clean', None) or chunk.content
                        texts_to_embed.append(content_to_embed)
                    
                    # Generate embeddings in batches
                    total_batches = (len(texts_to_embed) + batch_size - 1) // batch_size
                    
                    for i in range(0, len(texts_to_embed), batch_size):
                        batch_texts = texts_to_embed[i:i + batch_size]
                        batch_num = i // batch_size + 1
                        print(f"🔄 Processing embedding batch {batch_num}/{total_batches}")
                        
                        batch_embeddings = embedding_generator.generate_embeddings_batch(batch_texts)
                        embeddings.extend(batch_embeddings)
                    
                    success_count = sum(1 for emb in embeddings if emb is not None)
                    print(f"✅ Generated {success_count}/{len(chunks)} embeddings successfully")
                else:
                    # No embeddings requested, fill with None values
                    embeddings = [None] * len(chunks)

                # 4. Prepare chunk data for bulk insert
                chunk_data_tuples = []
                for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_hash = calculate_chunk_hash(chunk.content, chunk_index, doc_version_id)
                    char_count = len(chunk.content)
                    
                    chunk_data_tuples.append((
                        doc_version_id,      # document_version_id
                        chunk_index,         # chunk_index
                        chunk.content,       # text
                        chunk.content,       # text_clean (same as text for now)
                        chunk_hash,          # chunk_hash
                        chunk.page_start,    # page_start
                        chunk.page_end,      # page_end
                        None,                # heading_path (can be added later)
                        char_count,          # char_count
                        embedding            # embedding
                    ))

                # 5. Bulk insert chunks with embeddings
                extras.execute_values(
                    cur,
                    """
                    INSERT INTO chunks (
                        document_version_id, chunk_index, text, text_clean, 
                        chunk_hash, page_start, page_end, heading_path, char_count, embedding
                    )
                    VALUES %s;
                    """,
                    chunk_data_tuples,
                    template=None,
                    page_size=100
                )
                
                embedded_count = sum(1 for _, _, _, _, _, _, _, _, _, embedding in chunk_data_tuples if embedding is not None)
                print(f"✅ Bulk-inserted {len(chunk_data_tuples)} chunks ({embedded_count} with embeddings).")

                # 6. Update document version status
                cur.execute(
                    """
                    UPDATE document_versions 
                    SET processing_status = 'completed' 
                    WHERE id = %s;
                    """,
                    (doc_version_id,)
                )
                print(f"✅ Document processing completed for version ID {doc_version_id}")

        return True

    except psycopg2.Error as e:
        print(f"🔴 Database error occurred: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"🔴 An unexpected error occurred: {e}")
        if conn:
            conn.rollback()
        return False

def load_document_from_file(
    conn, 
    file_path: str, 
    chunks: List[TextChunk],
    embedding_model: str = "multi-qa-MiniLM-L6-cos-v1",
    batch_size: int = 32
) -> bool:
    """
    Convenience function to load a document from a file path with embedding generation.
    Automatically calculates file size and hash.
    
    Args:
        conn: Database connection
        file_path: Path to the source file
        chunks: List of TextChunk objects
        embedding_model: Name of the embedding model to use
        batch_size: Batch size for embedding generation
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"🔴 File not found: {file_path}")
        return False
    
    file_size, file_hash = get_file_info(file_path)
    if file_hash is None:
        return False
    
    # Calculate page count from chunks
    page_count = max(chunk.page_end for chunk in chunks) if chunks else 0
    
    return load_document_to_db(
        conn=conn,
        source_uri=file_path,
        source_hash=file_hash,
        chunks=chunks,
        file_size=file_size,
        page_count=page_count,
        embedding_model=embedding_model,
        batch_size=batch_size
    )
