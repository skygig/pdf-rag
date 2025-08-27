# local_similarity_search.py
# Search for similar text chunks using locally generated embeddings

import numpy as np
import json
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import for local embedding generation (for query embeddings)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available for query embedding!")

# Global model cache to avoid reloading
_model_cache = {}

def parse_embedding(embedding_data) -> Optional[List[float]]:
    """
    Parse embedding data from database, handling different storage formats.
    
    Args:
        embedding_data: Raw embedding data from database
        
    Returns:
        List of floats or None if parsing fails
    """
    try:
        if embedding_data is None:
            return None
        
        # If it's already a list
        if isinstance(embedding_data, list):
            return [float(x) for x in embedding_data]
        
        # If it's a string (JSON format)
        if isinstance(embedding_data, str):
            # Try to parse as JSON
            try:
                parsed = json.loads(embedding_data)
                if isinstance(parsed, list):
                    return [float(x) for x in parsed]
            except json.JSONDecodeError:
                # Try to parse as comma-separated values
                try:
                    # Remove brackets if present
                    clean_str = embedding_data.strip('[]{}()')
                    values = clean_str.split(',')
                    return [float(x.strip()) for x in values if x.strip()]
                except (ValueError, AttributeError):
                    pass
        
        # If it's a numpy array
        if isinstance(embedding_data, np.ndarray):
            return embedding_data.flatten().tolist()
        
        # If it's some other iterable
        try:
            return [float(x) for x in embedding_data]
        except (TypeError, ValueError):
            pass
        
        logger.error(f"Could not parse embedding data of type: {type(embedding_data)}")
        return None
        
    except Exception as e:
        logger.error(f"Error parsing embedding: {e}")
        return None

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        # Ensure both vectors are the same length
        if len(vec1) != len(vec2):
            logger.error(f"Vector length mismatch: {len(vec1)} vs {len(vec2)}")
            return 0.0
        
        # Convert to numpy arrays with explicit float type
        a = np.array(vec1, dtype=np.float32)  # Use float32 for memory efficiency
        b = np.array(vec2, dtype=np.float32)
        
        # Calculate cosine similarity using vectorized operations
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
        
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0

def get_query_embedding_local(query: str, model_name: str = "multi-qa-MiniLM-L6-cos-v1") -> Optional[List[float]]:
    """Generate embedding for a search query using local model with caching."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("sentence-transformers not available for query embedding!")
        return None
    
    try:
        # Check if model is already loaded in cache
        if model_name not in _model_cache:
            logger.info(f"Loading model: {model_name}")
            _model_cache[model_name] = SentenceTransformer(model_name)
        
        model = _model_cache[model_name]
        
        # Generate embedding
        embedding = model.encode(query.strip(), convert_to_tensor=False, normalize_embeddings=True)
        
        # Convert to list with consistent precision
        if isinstance(embedding, np.ndarray):
            embedding = embedding.astype(np.float32).tolist()
        
        logger.info(f"Generated query embedding (dimension: {len(embedding)})")
        return embedding
        
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        return None

def search_similar_chunks_with_pgvector(
    conn, 
    query: str, 
    limit: int = 5, 
    similarity_threshold: float = 0.7,
    model_name: str = "multi-qa-MiniLM-L6-cos-v1"
) -> List[Dict]:
    """
    Search using pgvector extension (if available).
    This is the fastest method for large datasets.
    """
    
    # Generate embedding for the query
    query_embedding = get_query_embedding_local(query, model_name)
    if not query_embedding:
        return []
    
    try:
        with conn.cursor() as cur:
            # Check if pgvector is available
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
            if not cur.fetchone():
                raise Exception("pgvector extension not found")
            
            # Convert embedding to string format for pgvector
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Use pgvector for similarity search with proper parameterization
            search_query = """
            SELECT 
                c.id,
                c.chunk_index,
                c.text,
                c.page_start,
                c.page_end,
                c.char_count,
                d.title,
                d.source_path,
                1 - (c.embedding <-> %s::vector) as similarity
            FROM chunks c
            JOIN document_versions dv ON c.document_version_id = dv.id
            JOIN documents d ON dv.document_id = d.id
            WHERE c.embedding IS NOT NULL
              AND 1 - (c.embedding <-> %s::vector) >= %s
            ORDER BY c.embedding <-> %s::vector
            LIMIT %s;
            """
            
            cur.execute(search_query, (
                embedding_str, embedding_str, similarity_threshold, embedding_str, limit
            ))
            
            results = cur.fetchall()
            
            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    'chunk_id': row[0],
                    'chunk_index': row[1],
                    'text': row[2],
                    'page_start': row[3],
                    'page_end': row[4],
                    'char_count': row[5],
                    'document_title': row[6],
                    'document_path': row[7],
                    'similarity_score': float(row[8]),
                    'search_method': 'pgvector'
                })
            
            logger.info(f"Found {len(results)} results using pgvector")
            return formatted_results
            
    except Exception as e:
        logger.warning(f"pgvector search failed: {e}")
        return []

def search_similar_chunks_manual(
    conn, 
    query: str, 
    limit: int = 5, 
    similarity_threshold: float = 0.7,
    model_name: str = "multi-qa-MiniLM-L6-cos-v1",
    batch_size: int = 1000
) -> List[Dict]:
    """
    Manual similarity search without pgvector with optimized batch processing.
    Slower but works on any PostgreSQL installation.
    """
    
    # Generate embedding for the query
    query_embedding = get_query_embedding_local(query, model_name)
    if not query_embedding:
        return []
    
    try:
        with conn.cursor() as cur:
            # Get total count first
            cur.execute("""
                SELECT COUNT(*) 
                FROM chunks c
                JOIN document_versions dv ON c.document_version_id = dv.id
                WHERE c.embedding IS NOT NULL
            """)
            total_count = cur.fetchone()[0]
            
            if total_count == 0:
                logger.warning("No chunks with embeddings found")
                return []
            
            logger.info(f"Processing {total_count} chunks for similarity calculation")
            
            # Process in batches for memory efficiency
            similarities = []
            processed = 0
            
            # Use cursor with name for server-side processing
            cur.execute("""
                DECLARE chunk_cursor CURSOR FOR
                SELECT 
                    c.id, c.chunk_index, c.text, c.page_start, c.page_end, 
                    c.char_count, c.embedding, d.title, d.source_path
                FROM chunks c
                JOIN document_versions dv ON c.document_version_id = dv.id
                JOIN documents d ON dv.document_id = d.id
                WHERE c.embedding IS NOT NULL
            """)
            
            while processed < total_count:
                cur.execute(f"FETCH {batch_size} FROM chunk_cursor")
                batch_results = cur.fetchall()
                
                if not batch_results:
                    break
                
                # Process batch
                for row in batch_results:
                    chunk_embedding_raw = row[6]  # embedding column
                    
                    # Parse the embedding from database
                    chunk_embedding = parse_embedding(chunk_embedding_raw)
                    
                    if chunk_embedding is None:
                        continue
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity(query_embedding, chunk_embedding)
                    
                    if similarity >= similarity_threshold:
                        similarities.append({
                            'chunk_id': row[0],
                            'chunk_index': row[1],
                            'text': row[2],
                            'page_start': row[3],
                            'page_end': row[4],
                            'char_count': row[5],
                            'document_title': row[7],
                            'document_path': row[8],
                            'similarity_score': similarity,
                            'search_method': 'manual'
                        })
                
                processed += len(batch_results)
                if processed % (batch_size * 5) == 0:  # Progress every 5 batches
                    logger.info(f"Processed {processed}/{total_count} chunks")
            
            # Close cursor
            cur.execute("CLOSE chunk_cursor")
            
            logger.info(f"Similarity calculation completed: {len(similarities)} results above threshold")
            
            # Sort by similarity (descending) and limit results
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similarities[:limit]
            
    except Exception as e:
        logger.error(f"Manual search failed: {e}")
        return []

def search_similar_chunks(
    conn, 
    query: str, 
    limit: int = 5, 
    similarity_threshold: float = 0.7,
    model_name: str = "multi-qa-MiniLM-L6-cos-v1"
) -> List[Dict]:
    """
    Smart similarity search that tries pgvector first, falls back to manual search.
    """
    
    logger.info(f"Searching for: '{query}'")
    logger.info(f"Parameters: limit={limit}, threshold={similarity_threshold}")
    
    # Validate inputs
    if not query.strip():
        logger.error("Empty query provided")
        return []
    
    if limit <= 0:
        logger.error("Limit must be positive")
        return []
    
    if not (0.0 <= similarity_threshold <= 1.0):
        logger.error("Similarity threshold must be between 0.0 and 1.0")
        return []
    
    # Try pgvector first (fastest)
    results = search_similar_chunks_with_pgvector(conn, query, limit, similarity_threshold, model_name)
    
    if results:
        logger.info(f"Found {len(results)} results using pgvector")
        return results
    
    # Fall back to manual search
    logger.info("Falling back to manual similarity calculation...")
    results = search_similar_chunks_manual(conn, query, limit, similarity_threshold, model_name)
    
    if results:
        logger.info(f"Found {len(results)} results using manual search")
        return results
    
    # Final fallback to text search
    logger.info("Falling back to text search...")
    return search_with_text_fallback(conn, query, limit)

def search_with_text_fallback(conn, query: str, limit: int = 5) -> List[Dict]:
    """Fallback to PostgreSQL full-text search if embedding search fails."""
    try:
        with conn.cursor() as cur:
            # Use parameterized query to prevent SQL injection
            search_query = """
            SELECT 
                c.id, c.chunk_index, c.text, c.page_start, c.page_end,
                c.char_count, d.title, d.source_path,
                ts_rank(to_tsvector('english', c.text), plainto_tsquery('english', %s)) as rank
            FROM chunks c
            JOIN document_versions dv ON c.document_version_id = dv.id
            JOIN documents d ON dv.document_id = d.id
            WHERE to_tsvector('english', c.text) @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank(to_tsvector('english', c.text), plainto_tsquery('english', %s)) DESC
            LIMIT %s;
            """
            
            cur.execute(search_query, (query, query, query, limit))
            results = cur.fetchall()
            
            formatted_results = []
            for row in results:
                formatted_results.append({
                    'chunk_id': row[0],
                    'chunk_index': row[1],
                    'text': row[2],
                    'page_start': row[3],
                    'page_end': row[4],
                    'char_count': row[5],
                    'document_title': row[6],
                    'document_path': row[7],
                    'text_rank': float(row[8]),
                    'search_method': 'text_search'
                })
            
            logger.info(f"Found {len(results)} results using text search")
            return formatted_results
            
    except Exception as e:
        logger.error(f"Text search also failed: {e}")
        return []

def display_search_results(results: List[Dict], query: str):
    """Display search results in a formatted way."""
    
    if not results:
        print(f"\n🔍 No results found for query: '{query}'")
        print("💡 Try:")
        print("   - Using different keywords")
        print("   - Lowering the similarity threshold")
        print("   - Checking if embeddings were generated correctly")
        return
    
    print(f"\n🔍 Search Results for: '{query}'")
    print(f"📊 Found {len(results)} results")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n📄 Result {i}")
        print(f"Document: {result.get('document_title', 'Unknown')}")
        print(f"Pages: {result.get('page_start', 'N/A')}-{result.get('page_end', 'N/A')}")
        print(f"Chunk ID: {result.get('chunk_id', 'N/A')} (Index: {result.get('chunk_index', 'N/A')})")
        print(f"Method: {result.get('search_method', 'unknown')}")
        
        # Show similarity score or text rank
        if 'similarity_score' in result:
            print(f"Similarity: {result['similarity_score']:.3f}")
        elif 'text_rank' in result:
            print(f"Text Rank: {result['text_rank']:.3f}")
        
        # Show text preview (first 300 characters)
        text_content = result.get('text', '')
        if len(text_content) > 300:
            text_preview = text_content[:300] + "..."
        else:
            text_preview = text_content
        
        print(f"Text: {text_preview}")
        print("-" * 80)

def check_embedding_status(conn):
    """Check the status of embeddings in the database."""
    try:
        with conn.cursor() as cur:
            # Get embedding statistics
            cur.execute("SELECT COUNT(*) FROM chunks;")
            total_chunks = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL;")
            embedded_chunks = cur.fetchone()[0]
            
            print(f"📊 Embedding Status:")
            print(f"   Total chunks: {total_chunks}")
            print(f"   With embeddings: {embedded_chunks}")
            
            if total_chunks > 0:
                coverage = (embedded_chunks / total_chunks * 100)
                print(f"   Coverage: {coverage:.1f}%")
            else:
                print("   Coverage: 0%")
            
            if embedded_chunks == 0:
                print("\n⚠️ No embeddings found!")
                print("Please run the embedding generation first.")
                return False
            
            # Check embedding format and dimensions
            cur.execute("SELECT embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 1;")
            sample_result = cur.fetchone()
            if sample_result:
                sample_embedding = sample_result[0]
                parsed_embedding = parse_embedding(sample_embedding)
                if parsed_embedding:
                    print(f"   Embedding dimension: {len(parsed_embedding)}")
                    print(f"   Embedding type: {type(sample_embedding)}")
                else:
                    print("   ⚠️ Could not parse sample embedding!")
                    print(f"   Raw embedding type: {type(sample_embedding)}")
                    if hasattr(sample_embedding, '__len__'):
                        print(f"   Raw embedding length: {len(sample_embedding)}")
            
            return True
            
    except Exception as e:
        logger.error(f"Error checking embedding status: {e}")
        return False

def test_embedding_parsing(conn):
    """Test embedding parsing for debugging."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 3;")
            results = cur.fetchall()
            
            print(f"\n🔧 Testing embedding parsing for {len(results)} samples:")
            
            for chunk_id, embedding_raw in results:
                print(f"\nChunk {chunk_id}:")
                print(f"  Raw type: {type(embedding_raw)}")
                
                # Show sample of raw data safely
                try:
                    raw_str = str(embedding_raw)
                    print(f"  Raw sample: {raw_str[:100]}{'...' if len(raw_str) > 100 else ''}")
                except Exception as e:
                    print(f"  Could not convert to string: {e}")
                
                parsed = parse_embedding(embedding_raw)
                if parsed:
                    print(f"  ✅ Parsed successfully: {len(parsed)} dimensions")
                    print(f"  Sample values: {parsed[:5]}...")
                else:
                    print(f"  ❌ Failed to parse")
            
    except Exception as e:
        logger.error(f"Error testing embedding parsing: {e}")

def interactive_search(conn, model_name: str = "multi-qa-MiniLM-L6-cos-v1"):
    """Interactive search interface with improved error handling."""
    print("🔍 Interactive Local Similarity Search")
    print("Type 'quit' to exit, 'help' for commands, 'status' to check embeddings")
    print("Type 'test' to test embedding parsing\n")
    
    while True:
        try:
            query = input("Enter search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif query.lower() == 'help':
                print("\nCommands:")
                print("  help   - Show this help")
                print("  status - Check embedding status")
                print("  test   - Test embedding parsing")
                print("  quit   - Exit the program")
                print("  Any other text - Search for similar chunks")
                print("\nExample queries:")
                print("  'solar system planets'")
                print("  'Jupiter Saturn gas giants'")
                print("  'asteroid belt Mars'")
                print("  'gravitational collapse'\n")
                continue
            elif query.lower() == 'status':
                check_embedding_status(conn)
                continue
            elif query.lower() == 'test':
                test_embedding_parsing(conn)
                continue
            elif not query:
                continue
            
            # Get search parameters with better validation
            try:
                limit_input = input("Number of results (default 5): ").strip()
                limit = int(limit_input) if limit_input else 5
                if limit <= 0:
                    print("Limit must be positive, using default value 5")
                    limit = 5
                    
                threshold_input = input("Similarity threshold 0-1 (default 0.5): ").strip()
                threshold = float(threshold_input) if threshold_input else 0.5
                if not (0.0 <= threshold <= 1.0):
                    print("Threshold must be between 0.0 and 1.0, using default 0.5")
                    threshold = 0.5
                    
            except ValueError:
                limit, threshold = 5, 0.5
                print("Invalid input, using default values: limit=5, threshold=0.5")
            
            print(f"\n🔄 Searching with model: {model_name}")
            
            # Perform search
            results = search_similar_chunks(conn, query, limit, threshold, model_name)
            
            # Display results
            display_search_results(results, query)
            
            print("\n" + "="*80 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error during search: {e}")
            print(f"🔴 Error during search: {e}")

def quick_search(conn, query: str, limit: int = 5, threshold: float = 0.7):
    """Quick search function for programmatic use with validation."""
    if not query.strip():
        logger.error("Empty query provided")
        return []
    
    results = search_similar_chunks(conn, query, limit, threshold)
    return results