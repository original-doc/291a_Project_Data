"""
Utility functions for PyTorch Lightning RAG System
Handles data loading, configuration parsing, and common operations
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a chunk of source code with metadata"""
    id: str
    text: str
    code: str
    documentation: str
    class_name: Optional[str] = None
    method_name: Optional[str] = None
    file_path: Optional[str] = None
    parent_index: Optional[int] = None
    chunk_type: str = "code"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocChunk:
    """Represents a chunk of documentation"""
    id: str
    text: str
    source_file: str
    section: Optional[str] = None
    chunk_type: str = "documentation"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscussionChunk:
    """Represents a GitHub discussion/issue"""
    id: str
    title: str
    body: str
    answer: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    chunk_type: str = "discussion"
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(__file__).parent.parent / config_path
    
    if not config_file.exists():
        # Try relative to current working directory
        config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_file}")
    return config


def get_data_path(config: Dict[str, Any], data_type: str) -> Path:
    """Get the full path for a specific data type"""
    base_path = Path(config['data']['base_path'])
    
    if data_type == "src_data":
        return base_path / config['data']['src_data']
    elif data_type == "docs":
        return base_path / config['data']['docs']
    elif data_type == "discussion":
        return base_path / config['data']['discussion']
    elif data_type == "request":
        return base_path / config['data']['request_file']
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def load_json_files(directory: Path) -> List[Dict[str, Any]]:
    """Load all JSON and JSONL files from a directory"""
    data = []
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return data
    
    # 1. Load standard JSON files
    for file_path in directory.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    for item in content:
                        item['_source_file'] = str(file_path.name)
                    data.extend(content)
                else:
                    content['_source_file'] = str(file_path.name)
                    data.append(content)
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")

    # 2. Load JSONL files (Output from the Dataset Builder)
    for file_path in directory.glob("*.jsonl"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            item['_source_file'] = str(file_path.name)
                            data.append(item)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Error loading JSONL file {file_path}: {e}")
    
    logger.info(f"Loaded {len(data)} items from {directory}")
    return data


def load_python_files(directory: Path) -> List[Dict[str, Any]]:
    """
    Recursively load all Python files from a directory.
    Includes filtering logic inspired by the Dataset Builder to avoid noise.
    """
    data = []
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return data

    # Exclusion patterns (Learned from Dataset Builder)
    exclude_patterns = [
        '.git', '__pycache__', '.egg-info', 'node_modules', 
        '.tox', 'build', 'dist', 'setup.py'
    ]
    # Optionally exclude tests if you want a cleaner RAG knowledge base
    exclude_tests = True 

    logger.info(f"Scanning for .py files in {directory}...")

    # Use rglob to find .py files recursively
    for file_path in directory.rglob("*.py"):
        path_str = str(file_path)
        
        # Check exclusion patterns
        if any(ex in path_str for ex in exclude_patterns):
            continue
            
        if exclude_tests and ('test' in path_str.lower() or 'tests' in path_str.lower()):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
                # Construct a dictionary compatible with the pipeline
                data.append({
                    'code': content,
                    'text': content, 
                    'file_path': str(file_path),
                    '_source_file': file_path.name,
                    # Metadata placeholders
                    'class_name': None,
                    'method_name': None,
                    'docstring': '',
                    'func_name': None
                })
        except Exception as e:
            logger.error(f"Error loading Python file {file_path}: {e}")
            
    logger.info(f"Loaded {len(data)} Python files from {directory} (after filtering)")
    return data


def load_src_data(config: Dict[str, Any]) -> List[CodeChunk]:
    """
    Load and parse source code data.
    Supports:
    1. Pre-processed JSON/JSONL files (from Dataset Builder)
    2. Raw .py files (using recursive scanner)
    """
    path = get_data_path(config, "src_data")
    
    # 1. Try loading pre-processed data (JSON/JSONL)
    raw_data = load_json_files(path)
    
    # 2. If no data found, assume raw source directory and scan .py files
    if not raw_data:
        logger.info(f"No JSON/JSONL files found in {path}. Scanning for raw .py files...")
        raw_data = load_python_files(path)
    
    chunks = []
    for idx, item in enumerate(raw_data):
        # MAPPING LOGIC: Supports both Builder output and legacy format
        
        # 1. Code Content
        code = item.get('code', item.get('Code', ''))
        
        # 2. Documentation (Builder uses 'docstring', legacy uses 'Documentation')
        doc = item.get('docstring', item.get('Documentation', item.get('Class Description', '')))
        
        # 3. Class Name
        class_name = item.get('class_name', item.get('Class', None))
        
        # 4. Method Name (Builder uses 'func_name' often for qualified names, or extract from name)
        method_name = item.get('method_name', item.get('Method', item.get('func_name', None)))
        
        # 5. File Path
        file_path = item.get('path', item.get('Path', item.get('file_path', None)))

        chunk = CodeChunk(
            id=f"code_{idx}",
            text=item.get('text', code), # Fallback to code if no specific text
            code=code,
            documentation=doc,
            class_name=class_name,
            method_name=method_name,
            file_path=file_path,
            parent_index=item.get('parent_index', None),
            metadata={
                'source_file': item.get('_source_file', ''),
                'original_index': idx,
                'is_synthetic': item.get('is_synthetic_docstring', False),
                'complexity': item.get('complexity', 0)
            }
        )
        chunks.append(chunk)
    
    logger.info(f"Loaded {len(chunks)} source code chunks")
    return chunks


def load_docs_data(config: Dict[str, Any]) -> List[DocChunk]:
    """Load and parse documentation data"""
    path = get_data_path(config, "docs")
    raw_data: List[Dict[str, Any]] = []

    if path.is_dir():
        raw_data = load_json_files(path)
    elif path.is_file():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)

            if isinstance(content, list):
                for item in content:
                    item.setdefault('_source_file', path.name)
                raw_data = content
            else:
                content.setdefault('_source_file', path.name)
                raw_data = [content]
            logger.info(f"Loaded {len(raw_data)} documentation items from file {path}")
        except Exception as e:
            logger.error(f"Error loading documentation file {path}: {e}")
    else:
        logger.warning(f"Documentation path does not exist: {path}")
    
    chunks = []
    for idx, item in enumerate(raw_data):
        chunk = DocChunk(
            id=f"doc_{idx}",
            text=item.get('text', item.get('content', '')),
            source_file=item.get('_source_file', ''),
            section=item.get('section', item.get('title', None)),
            metadata={
                'original_index': idx,
                'url': item.get('url', ''),
                'category': item.get('category', '')
            }
        )
        chunks.append(chunk)
    
    logger.info(f"Loaded {len(chunks)} documentation chunks")
    return chunks


def load_discussion_data(config: Dict[str, Any]) -> List[DiscussionChunk]:
    """Load and parse GitHub discussion data"""
    path = get_data_path(config, "discussion")
    raw_data: List[Dict[str, Any]] = []

    if path.is_dir():
        raw_data = load_json_files(path)
    elif path.is_file():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)

            if isinstance(content, list):
                for item in content:
                    item.setdefault('_source_file', path.name)
                raw_data = content
            else:
                content.setdefault('_source_file', path.name)
                raw_data = [content]
            logger.info(f"Loaded {len(raw_data)} discussion items from file {path}")
        except Exception as e:
            logger.error(f"Error loading discussion file {path}: {e}")
    else:
        logger.warning(f"Discussion path does not exist: {path}")
    
    chunks = []
    for idx, item in enumerate(raw_data):
        chunk = DiscussionChunk(
            id=f"discussion_{idx}",
            title=item.get('title', ''),
            body=item.get('bodyText', item.get('body', '')),
            answer=item.get('answer', item.get('top_answer', None)),
            labels=item.get('labels', []),
            metadata={
                'source_file': item.get('_source_file', ''),
                'original_index': idx,
                'url': item.get('url', ''),
                'created_at': item.get('created_at', ''),
                'author': item.get('author', '')
            }
        )
        chunks.append(chunk)
    
    logger.info(f"Loaded {len(chunks)} discussion chunks")
    return chunks


def load_request_data(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load evaluation request/query data"""
    path = get_data_path(config, "request")
    
    if not path.exists():
        logger.warning(f"Request file does not exist: {path}")
        return []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        logger.info(f"Loaded {len(data)} evaluation requests")
        return data
    except Exception as e:
        logger.error(f"Error loading request file: {e}")
        return []


def load_all_data(config: Dict[str, Any]) -> Dict[str, List]:
    """Load all data types"""
    return {
        'src_data': load_src_data(config),
        'docs': load_docs_data(config),
        'discussion': load_discussion_data(config)
    }


def chunk_to_dict(chunk: Union[CodeChunk, DocChunk, DiscussionChunk]) -> Dict[str, Any]:
    """Convert a chunk dataclass to dictionary"""
    if isinstance(chunk, CodeChunk):
        return {
            'id': chunk.id,
            'text': chunk.text,
            'code': chunk.code,
            'documentation': chunk.documentation,
            'class_name': chunk.class_name,
            'method_name': chunk.method_name,
            'file_path': chunk.file_path,
            'parent_index': chunk.parent_index,
            'chunk_type': chunk.chunk_type,
            'metadata': chunk.metadata
        }
    elif isinstance(chunk, DocChunk):
        return {
            'id': chunk.id,
            'text': chunk.text,
            'source_file': chunk.source_file,
            'section': chunk.section,
            'chunk_type': chunk.chunk_type,
            'metadata': chunk.metadata
        }
    elif isinstance(chunk, DiscussionChunk):
        return {
            'id': chunk.id,
            'title': chunk.title,
            'body': chunk.body,
            'answer': chunk.answer,
            'labels': chunk.labels,
            'chunk_type': chunk.chunk_type,
            'metadata': chunk.metadata
        }
    else:
        raise TypeError(f"Unknown chunk type: {type(chunk)}")


def get_chunk_text(chunk: Union[CodeChunk, DocChunk, DiscussionChunk]) -> str:
    """Extract the main text content from any chunk type"""
    if isinstance(chunk, CodeChunk):
        parts = []
        if chunk.documentation:
            parts.append(chunk.documentation)
        if chunk.code:
            parts.append(chunk.code)
        return "\n".join(parts) if parts else chunk.text
    elif isinstance(chunk, DocChunk):
        return chunk.text
    elif isinstance(chunk, DiscussionChunk):
        parts = [chunk.title, chunk.body]
        if chunk.answer:
            parts.append(chunk.answer)
        return "\n".join(parts)
    else:
        raise TypeError(f"Unknown chunk type: {type(chunk)}")


if __name__ == "__main__":
    try:
        config = load_config()
        print("Configuration loaded successfully")
        data = load_all_data(config)
        for key, items in data.items():
            print(f"{key}: {len(items)} items")
    except Exception as e:
        print(f"Error: {e}")