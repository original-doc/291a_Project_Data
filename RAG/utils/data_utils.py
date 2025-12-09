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
    """Load all JSON files from a directory"""
    data = []
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return data
    
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
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
    
    logger.info(f"Loaded {len(data)} items from {directory}")
    return data


def load_src_data(config: Dict[str, Any]) -> List[CodeChunk]:
    """Load and parse source code data"""
    path = get_data_path(config, "src_data")
    raw_data = load_json_files(path)
    
    chunks = []
    for idx, item in enumerate(raw_data):
        chunk = CodeChunk(
            id=f"code_{idx}",
            text=item.get('text', ''),
            code=item.get('Code', item.get('code', '')),
            documentation=item.get('Documentation', item.get('Class Description', '')),
            class_name=item.get('Class', item.get('class_name', None)),
            method_name=item.get('Method', item.get('method_name', None)),
            file_path=item.get('Path', item.get('file_path', None)),
            parent_index=item.get('parent_index', None),
            metadata={
                'source_file': item.get('_source_file', ''),
                'original_index': idx
            }
        )
        chunks.append(chunk)
    
    logger.info(f"Loaded {len(chunks)} source code chunks")
    return chunks


def load_docs_data(config: Dict[str, Any]) -> List[DocChunk]:
    """Load and parse documentation data"""
    path = get_data_path(config, "docs")
    raw_data: List[Dict[str, Any]] = []

    # Support both a directory of JSON files and a single JSON file
    if path.is_dir():
        raw_data = load_json_files(path)
    elif path.is_file():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)

            if isinstance(content, list):
                for item in content:
                    # Track where this item came from for downstream debugging
                    item.setdefault('_source_file', path.name)
                raw_data = content
            else:
                content.setdefault('_source_file', path.name)
                raw_data = [content]

            logger.info(f"Loaded {len(raw_data)} documentation items from file {path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing documentation JSON file {path}: {e}")
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

    # Support both a directory of JSON files and a single JSON file
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
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing discussion JSON file {path}: {e}")
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
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    
    logger.info(f"Loaded {len(data)} evaluation requests")
    return data


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
        # Combine code and documentation for embedding
        parts = []
        if chunk.documentation:
            parts.append(chunk.documentation)
        if chunk.code:
            parts.append(chunk.code)
        return "\n".join(parts) if parts else chunk.text
    elif isinstance(chunk, DocChunk):
        return chunk.text
    elif isinstance(chunk, DiscussionChunk):
        # Combine title, body, and answer
        parts = [chunk.title, chunk.body]
        if chunk.answer:
            parts.append(chunk.answer)
        return "\n".join(parts)
    else:
        raise TypeError(f"Unknown chunk type: {type(chunk)}")


if __name__ == "__main__":
    # Test data loading
    try:
        config = load_config()
        print("Configuration loaded successfully")
        
        data = load_all_data(config)
        for key, items in data.items():
            print(f"{key}: {len(items)} items")
    except Exception as e:
        print(f"Error: {e}")
