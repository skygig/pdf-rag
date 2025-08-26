# local_embedding_generator.py : Generate embeddings using free local models (sentence-transformers)

import numpy as np
from typing import List, Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("⚠️ sentence-transformers not installed. Install with: pip install sentence-transformers torch")


class LocalEmbeddingGenerator:
    """Generate embeddings using local sentence-transformer models."""
    
    def __init__(self, model_name: str = "multi-qa-MiniLM-L6-cos-v1"):
        """
        Initialize with a sentence-transformer model.
        
        Popular free models:
        - "all-MiniLM-L6-v2" (384 dims, fast, good quality)
        - "all-mpnet-base-v2" (768 dims, slower, better quality)  
        - "multi-qa-MiniLM-L6-cos-v1" (384 dims, optimized for Q&A)
        """

        print(f"🔄 Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text."""
        try:
            if not text.strip():
                return None
            
            # Generate embedding
            embedding = self.model.encode(text.strip(), convert_to_tensor=False)
            
            # Convert to list and ensure it's the right type
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            print(f"🔴 Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts efficiently."""
        try:
            # Filter out empty texts but keep track of indices
            text_mapping = []
            valid_texts = []
            
            for i, text in enumerate(texts):
                if text and text.strip():
                    text_mapping.append(i)
                    valid_texts.append(text.strip())
            
            if not valid_texts:
                return [None] * len(texts)
            
            # Generate embeddings in batch
            embeddings = self.model.encode(valid_texts, convert_to_tensor=False, show_progress_bar=True)
            
            # Map back to original indices
            result = [None] * len(texts)
            for i, embedding in enumerate(embeddings):
                original_index = text_mapping[i]
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                result[original_index] = embedding
            
            return result
            
        except Exception as e:
            print(f"🔴 Error generating batch embeddings: {e}")
            return [None] * len(texts)
