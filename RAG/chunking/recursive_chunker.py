"""
Recursive Chunking for Documentation

This module implements recursive text chunking for documentation files,
splitting by logical separators while preserving code blocks and
maintaining semantic coherence.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text from documentation"""
    id: str
    text: str
    source_file: str
    section_title: Optional[str] = None
    section_hierarchy: List[str] = field(default_factory=list)
    chunk_index: int = 0
    total_chunks: int = 0
    has_code: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecursiveTextChunker:
    """
    Recursive text chunker that splits documentation by logical separators.
    
    Uses a hierarchy of separators, starting with the most semantically
    meaningful (headers, section breaks) and falling back to less
    meaningful ones (paragraphs, sentences) as needed.
    """
    
    DEFAULT_SEPARATORS = [
        "\n## ",      # H2 headers (Markdown)
        "\n### ",     # H3 headers
        "\n#### ",    # H4 headers
        "\n--- ",     # Section breaks
        "```",        # Markdown code blocks
        "\n\n",       # Paragraphs
        "\n",         # Lines
        ". ",         # Sentences
        " ",          # Words (last resort)
    ]
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        preserve_code_blocks: bool = True,
        length_function: Optional[callable] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.preserve_code_blocks = preserve_code_blocks
        self.length_function = length_function or len
    
    def _extract_code_blocks(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Extract code blocks and replace with placeholders.
        
        Returns:
            Tuple of (text_with_placeholders, placeholder_map)
        """
        placeholders = {}
        placeholder_id = 0
        
        # Extract markdown code blocks
        def replace_markdown_code(match):
            nonlocal placeholder_id
            placeholder = f"__CODE_BLOCK_{placeholder_id}__"
            placeholders[placeholder] = match.group(0)
            placeholder_id += 1
            return placeholder
        
        # Match ```...``` blocks
        text = re.sub(
            r'```[\s\S]*?```',
            replace_markdown_code,
            text
        )
        
        # Match RST code blocks (.. code:: ... followed by indented content)
        text = re.sub(
            r'\.\. code::.*?\n(?:[ \t]+.*?\n)*',
            replace_markdown_code,
            text
        )
        
        return text, placeholders
    
    def _restore_code_blocks(self, text: str, placeholders: Dict[str, str]) -> str:
        """Restore code blocks from placeholders"""
        for placeholder, code in placeholders.items():
            text = text.replace(placeholder, code)
        return text
    
    def _extract_section_title(self, text: str) -> Optional[str]:
        """Extract the section title from the beginning of text"""
        # Check for markdown headers
        header_match = re.match(r'^#+\s+(.+?)(?:\n|$)', text.strip())
        if header_match:
            return header_match.group(1).strip()
        
        # Check for RST headers (text followed by === or ---)
        rst_match = re.match(r'^(.+?)\n[=\-]{3,}', text.strip())
        if rst_match:
            return rst_match.group(1).strip()
        
        return None
    
    def _split_by_separator(
        self,
        text: str,
        separator: str
    ) -> List[str]:
        """Split text by a separator, keeping the separator with the following chunk"""
        if separator == " ":
            # For word-level splitting, use simple split
            return text.split(separator)
        
        # For other separators, keep them with the following chunk
        parts = text.split(separator)
        
        if len(parts) <= 1:
            return parts
        
        # Add separator back to the beginning of each part (except first)
        result = [parts[0]]
        for part in parts[1:]:
            result.append(separator.lstrip('\n') + part)
        
        return result
    
    def _merge_small_chunks(
        self,
        chunks: List[str],
        separator: str
    ) -> List[str]:
        """Merge chunks that are smaller than chunk_size"""
        if not chunks:
            return chunks
        
        merged = []
        current = chunks[0]
        
        for chunk in chunks[1:]:
            # Check if adding this chunk would exceed the size limit
            combined = current + separator + chunk
            if self.length_function(combined) <= self.chunk_size:
                current = combined
            else:
                if current.strip():
                    merged.append(current)
                current = chunk
        
        if current.strip():
            merged.append(current)
        
        return merged
    
    def _recursive_split(
        self,
        text: str,
        separators: List[str]
    ) -> List[str]:
        """Recursively split text using the separator hierarchy"""
        if not text.strip():
            return []
        
        # If text is small enough, return it
        if self.length_function(text) <= self.chunk_size:
            return [text]
        
        # If no more separators, force split by character
        if not separators:
            # Hard split by chunk_size
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        splits = self._split_by_separator(text, separator)
        
        # If separator didn't split the text, try next separator
        if len(splits) == 1:
            return self._recursive_split(text, remaining_separators)
        
        # Process each split recursively
        chunks = []
        for split in splits:
            if self.length_function(split) <= self.chunk_size:
                if split.strip():
                    chunks.append(split)
            else:
                # Recursively split this chunk
                sub_chunks = self._recursive_split(split, remaining_separators)
                chunks.extend(sub_chunks)
        
        # Merge chunks that are too small
        chunks = self._merge_small_chunks(chunks, "\n")
        
        return chunks
    
    def chunk_text(
        self,
        text: str,
        source_file: str = "",
        base_id: str = "doc"
    ) -> List[TextChunk]:
        """
        Split text into chunks using recursive splitting.
        
        Args:
            text: The text to chunk
            source_file: Source file name for metadata
            base_id: Base ID for chunk IDs
            
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []
        
        # Extract and preserve code blocks if enabled
        if self.preserve_code_blocks:
            processed_text, placeholders = self._extract_code_blocks(text)
        else:
            processed_text = text
            placeholders = {}
        
        # Perform recursive splitting
        raw_chunks = self._recursive_split(processed_text, self.separators)
        
        # Create TextChunk objects
        chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            # Restore code blocks
            if placeholders:
                chunk_text = self._restore_code_blocks(chunk_text, placeholders)
            
            # Extract section title
            section_title = self._extract_section_title(chunk_text)
            
            # Check if chunk contains code
            has_code = bool(re.search(r'```|.. code::', chunk_text))
            
            chunk = TextChunk(
                id=f"{base_id}_{idx}",
                text=chunk_text.strip(),
                source_file=source_file,
                section_title=section_title,
                chunk_index=idx,
                total_chunks=len(raw_chunks),
                has_code=has_code,
                metadata={
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split())
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_with_overlap(
        self,
        text: str,
        source_file: str = "",
        base_id: str = "doc"
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks for better context preservation.
        """
        # First, get non-overlapping chunks
        chunks = self.chunk_text(text, source_file, base_id)
        
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks
        
        # Add overlap from previous chunk
        overlapped_chunks = []
        for idx, chunk in enumerate(chunks):
            if idx > 0:
                # Get overlap from previous chunk
                prev_text = chunks[idx - 1].text
                overlap_text = prev_text[-self.chunk_overlap:]
                
                # Add context marker
                new_text = f"[...] {overlap_text}\n{chunk.text}"
                chunk.text = new_text
                chunk.metadata['has_overlap'] = True
            
            overlapped_chunks.append(chunk)
        
        return overlapped_chunks


class DiscussionChunker:
    """
    Specialized chunker for GitHub discussions/issues.
    
    Treats problem (bodyText) and solution (answer) as paired units
    to preserve the Q&A relationship.
    """
    
    def __init__(self, max_length: int = 1024):
        self.max_length = max_length
        self.text_chunker = RecursiveTextChunker(chunk_size=max_length)
    
    def chunk_discussion(
        self,
        title: str,
        body: str,
        answer: Optional[str] = None,
        labels: Optional[List[str]] = None,
        discussion_id: str = "disc"
    ) -> List[TextChunk]:
        """
        Create chunks from a discussion, preserving Q&A pairing.
        
        Args:
            title: Discussion title
            body: Discussion body (the question/problem)
            answer: The answer/solution (if available)
            labels: Discussion labels/tags
            discussion_id: Base ID for chunks
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        
        # Create the combined Q&A text
        combined_parts = [f"# {title}", body]
        if answer:
            combined_parts.append(f"\n## Answer\n{answer}")
        
        combined_text = "\n\n".join(combined_parts)
        
        # Check if the combined text fits in one chunk
        if len(combined_text) <= self.max_length:
            chunk = TextChunk(
                id=f"{discussion_id}_qa",
                text=combined_text,
                source_file="discussion",
                section_title=title,
                has_code=bool(re.search(r'```|`[^`]+`', combined_text)),
                metadata={
                    'type': 'qa_pair',
                    'has_answer': answer is not None,
                    'labels': labels or []
                }
            )
            return [chunk]
        
        # If too long, create separate chunks but link them
        # Question chunk
        question_text = f"# {title}\n\n{body}"
        question_chunks = self.text_chunker.chunk_text(
            question_text, "discussion", f"{discussion_id}_q"
        )
        
        for chunk in question_chunks:
            chunk.metadata['type'] = 'question'
            chunk.metadata['labels'] = labels or []
            chunk.metadata['linked_answer'] = f"{discussion_id}_a" if answer else None
        
        chunks.extend(question_chunks)
        
        # Answer chunk (if present)
        if answer:
            answer_chunks = self.text_chunker.chunk_text(
                f"## Answer to: {title}\n\n{answer}",
                "discussion",
                f"{discussion_id}_a"
            )
            
            for chunk in answer_chunks:
                chunk.metadata['type'] = 'answer'
                chunk.metadata['linked_question'] = f"{discussion_id}_q_0"
            
            chunks.extend(answer_chunks)
        
        return chunks


def chunk_to_dict(chunk: TextChunk) -> Dict[str, Any]:
    """Convert TextChunk to dictionary for serialization"""
    return {
        'id': chunk.id,
        'text': chunk.text,
        'source_file': chunk.source_file,
        'section_title': chunk.section_title,
        'section_hierarchy': chunk.section_hierarchy,
        'chunk_index': chunk.chunk_index,
        'total_chunks': chunk.total_chunks,
        'has_code': chunk.has_code,
        'metadata': chunk.metadata
    }


if __name__ == "__main__":
    # Test the recursive chunker
    test_doc = """
# PyTorch Lightning Fabric

Fabric is a high-level interface for PyTorch Lightning that provides
a simple way to train models with minimal boilerplate code.

## Installation

You can install Fabric using pip:

```bash
pip install lightning
```

Or using conda:

```bash
conda install lightning -c conda-forge
```

## Quick Start

Here's a simple example of using Fabric:

```python
from lightning.fabric import Fabric

fabric = Fabric()
fabric.launch()

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())

model, optimizer = fabric.setup(model, optimizer)
```

## Configuration

Fabric supports various configuration options including:

--- Metadata ---
- Accelerator: auto, cpu, gpu, tpu
- Devices: auto, 1, 2, etc.
- Precision: 32, 16, bf16
"""
    
    chunker = RecursiveTextChunker(chunk_size=300)
    chunks = chunker.chunk_text(test_doc, "fabric_docs.md")
    
    print(f"Created {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"{'='*50}")
        print(f"ID: {chunk.id}")
        print(f"Section: {chunk.section_title}")
        print(f"Has code: {chunk.has_code}")
        print(f"Length: {len(chunk.text)} chars")
        print(f"Text preview: {chunk.text[:200]}...")
