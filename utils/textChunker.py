# text_chunker.py : It takes extracted text from a PDF and splits it into logical chunks.

import re
from typing import List, Tuple


class TextChunk:
    """A class to represent a single chunk of text."""
    def __init__(self, content: str, page_start: int, page_end: int):
        self.content = content
        self.page_start = page_start
        self.page_end = page_end

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the object."""
        content_preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return (f"TextChunk(page='{self.page_start}-{self.page_end}', "
                f"size={len(self.content)}, "
                f"content='{content_preview}')")

def chunk_text(
    pages: List[Tuple[int, str]],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[TextChunk]:
    """
    Splits the text from PDF pages into manageable chunks.

    Args:
        pages: A list of tuples, where each tuple contains a page number
               and the text content of that page.
        chunk_size: The target size for each chunk in characters.
        chunk_overlap: The number of characters to overlap between chunks
                       to maintain context.

    Returns:
        A list of TextChunk objects.
    """
    if not pages:
        return []
    
    chunks = []
    
    # First, combine all text into a single string while keeping track of page breaks.
    full_text = ""
    page_boundaries = []  # To track where each page starts and ends in the full text
    
    for page_num, page_text in pages:
        start_index = len(full_text)
        # Clean individual page text before adding
        cleaned_page_text = re.sub(r'\s+', ' ', page_text.strip())
        if full_text and not full_text.endswith(' '):
            full_text += " "
        full_text += cleaned_page_text
        end_index = len(full_text)
        page_boundaries.append((start_index, end_index, page_num))

    if not full_text:
        return []

    # Now, split the text into chunks
    start_index = 0
    
    while start_index < len(full_text):
        end_index = min(start_index + chunk_size, len(full_text))
        
        # Find the best place to split to avoid breaking sentences or words
        split_pos = end_index
        
        if end_index < len(full_text):  # Not at the end of text
            # Look for sentence endings first (period, exclamation, question mark followed by space)
            for punct in ['. ', '! ', '? ']:
                pos = full_text.rfind(punct, start_index, end_index)
                if pos > start_index:
                    split_pos = pos + 1  # Include the period, exclude the space
                    break
            
            # If no sentence ending found, look for other natural break points
            if split_pos == end_index:
                for break_char in ['; ', ', ', ' ']:
                    pos = full_text.rfind(break_char, start_index, end_index)
                    if pos > start_index:
                        split_pos = pos + (1 if break_char != ' ' else 0)
                        break
            
            # Ensure we don't break words - find the last space before our split point
            if split_pos == end_index and end_index < len(full_text):
                space_pos = full_text.rfind(' ', start_index, end_index)
                if space_pos > start_index:
                    split_pos = space_pos

        chunk_content = full_text[start_index:split_pos].strip()
        
        # Skip empty chunks
        if not chunk_content:
            start_index = split_pos + 1
            continue
        
        # Find the start and end pages for this chunk
        start_page = None
        end_page = None
        
        for boundary_start, boundary_end, page_num in page_boundaries:
            # Check if the chunk overlaps with this page's text range
            if start_index < boundary_end and split_pos > boundary_start:
                if start_page is None:
                    start_page = page_num
                end_page = page_num

        # Only create chunk if we found valid page numbers
        if start_page is not None:
            chunks.append(TextChunk(
                content=chunk_content,
                page_start=start_page,
                page_end=end_page
            ))
        
        # Calculate next start position with overlap
        if split_pos >= len(full_text):
            break
            
        # For overlap, go back from split_pos to find a good starting point
        overlap_start = max(start_index, split_pos - chunk_overlap)
        
        # Try to start the next chunk at a word boundary within the overlap region
        if overlap_start < split_pos:
            space_pos = full_text.find(' ', overlap_start)
            if space_pos != -1 and space_pos < split_pos:
                next_start = space_pos + 1
            else:
                next_start = overlap_start
        else:
            next_start = split_pos
            
        # Ensure we make progress
        if next_start <= start_index:
            next_start = start_index + 1
        
        start_index = next_start

    return chunks
