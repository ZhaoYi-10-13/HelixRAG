# Copyright 2024
# Directory: Gary-Agent-RAG/app/services/document_parser.py

"""
Enhanced document parsing service using LlamaIndex for better document processing.
Supports multiple file formats with intelligent parsing and metadata extraction.
"""

import logging
import os
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.schema import TextNode
from llama_index.readers.file import PDFReader, DocxReader, UnstructuredReader
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentParser:
    """Enhanced document parser using LlamaIndex for better file processing."""
    
    def __init__(self):
        """Initialize document parser with various readers."""
        self.readers = {
            '.pdf': PDFReader(),
            '.docx': DocxReader(),
            '.txt': UnstructuredReader(),
            '.md': UnstructuredReader(),
            '.html': UnstructuredReader(),
            '.csv': UnstructuredReader(),
        }
        logger.info("Initialized enhanced document parser with LlamaIndex")
    
    def parse_file(self, file_path: str, file_type: str = None) -> List[Document]:
        """
        Parse a single file using appropriate reader.
        
        Args:
            file_path: Path to the file to parse
            file_type: Optional file type override
            
        Returns:
            List of parsed documents
        """
        try:
            file_path = Path(file_path)
            
            # Determine file type
            if file_type:
                ext = f".{file_type.lower()}"
            else:
                ext = file_path.suffix.lower()
            
            # Get appropriate reader
            reader = self.readers.get(ext)
            if not reader:
                logger.warning(f"No reader available for file type: {ext}")
                # Fallback to unstructured reader
                reader = UnstructuredReader()
            
            # Parse the file
            documents = reader.load_data(file_path)
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'file_type': ext,
                    'file_size': file_path.stat().st_size if file_path.exists() else 0
                })
            
            logger.info(f"Parsed file {file_path.name}: {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {e}")
            return []
    
    def parse_files(self, file_paths: List[str]) -> List[Document]:
        """
        Parse multiple files.
        
        Args:
            file_paths: List of file paths to parse
            
        Returns:
            List of all parsed documents
        """
        all_documents = []
        
        for file_path in file_paths:
            documents = self.parse_file(file_path)
            all_documents.extend(documents)
        
        logger.info(f"Parsed {len(file_paths)} files: {len(all_documents)} total documents")
        return all_documents
    
    def parse_directory(self, directory_path: str, file_extensions: List[str] = None) -> List[Document]:
        """
        Parse all files in a directory.
        
        Args:
            directory_path: Path to directory
            file_extensions: Optional list of file extensions to include
            
        Returns:
            List of all parsed documents
        """
        try:
            directory_path = Path(directory_path)
            
            if file_extensions:
                # Filter by extensions
                file_paths = []
                for ext in file_extensions:
                    if not ext.startswith('.'):
                        ext = f'.{ext}'
                    file_paths.extend(directory_path.glob(f'**/*{ext}'))
            else:
                # Use all supported file types
                supported_extensions = list(self.readers.keys())
                file_paths = []
                for ext in supported_extensions:
                    file_paths.extend(directory_path.glob(f'**/*{ext}'))
            
            return self.parse_files([str(fp) for fp in file_paths])
            
        except Exception as e:
            logger.error(f"Failed to parse directory {directory_path}: {e}")
            return []
    
    def parse_uploaded_files(self, uploaded_files: List[Any]) -> List[Document]:
        """
        Parse uploaded files (from web upload).
        
        Args:
            uploaded_files: List of uploaded file objects
            
        Returns:
            List of parsed documents
        """
        all_documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.filename).suffix) as tmp_file:
                    # Write uploaded content to temp file
                    content = uploaded_file.file.read()
                    tmp_file.write(content)
                    tmp_file_path = tmp_file.name
                
                # Parse the temporary file
                documents = self.parse_file(tmp_file_path)
                all_documents.extend(documents)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
            except Exception as e:
                logger.error(f"Failed to parse uploaded file {uploaded_file.filename}: {e}")
                continue
        
        logger.info(f"Parsed {len(uploaded_files)} uploaded files: {len(all_documents)} total documents")
        return all_documents
    
    def convert_to_chunks(self, documents: List[Document], chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
        """
        Convert documents to chunks for RAG processing.
        
        Args:
            documents: List of documents to chunk
            chunk_size: Chunk size override
            overlap: Overlap size override
            
        Returns:
            List of chunk dictionaries
        """
        chunk_size = chunk_size or settings.chunk_size
        overlap = overlap or settings.chunk_overlap
        
        chunks = []
        
        for doc in documents:
            # Use LlamaIndex's built-in chunking
            text_nodes = self._chunk_document(doc, chunk_size, overlap)
            
            for i, node in enumerate(text_nodes):
                chunk = {
                    'chunk_id': f"{doc.metadata.get('file_name', 'doc')}#{i+1}",
                    'source': doc.metadata.get('file_path', 'unknown'),
                    'text': node.text,
                    'metadata': {
                        **doc.metadata,
                        'chunk_index': i,
                        'total_chunks': len(text_nodes)
                    }
                }
                chunks.append(chunk)
        
        logger.info(f"Converted {len(documents)} documents to {len(chunks)} chunks")
        return chunks
    
    def _chunk_document(self, document: Document, chunk_size: int, overlap: int) -> List[TextNode]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            document: Document to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text nodes
        """
        text = document.text
        words = text.split()
        
        chunks = []
        start_idx = 0
        chunk_num = 1
        
        while start_idx < len(words):
            end_idx = min(start_idx + chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            # Create TextNode
            node = TextNode(text=chunk_text)
            node.metadata = {
                **document.metadata,
                'chunk_id': f"{document.metadata.get('file_name', 'doc')}#{chunk_num}",
                'chunk_index': chunk_num - 1
            }
            chunks.append(node)
            
            if end_idx >= len(words):
                break
            
            start_idx = end_idx - overlap
            chunk_num += 1
        
        return chunks


# Global document parser instance
document_parser = DocumentParser()
