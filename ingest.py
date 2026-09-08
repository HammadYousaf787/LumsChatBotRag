"""
Document Ingestion Pipeline for LUMS Academic Knowledge Assistant
Supports incremental loading - only processes new PDFs
"""

import os
import logging
import json
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class DocumentIngestor:
    """Handles incremental ingestion of PDF documents into the vector database."""
    
    def __init__(self, data_dir: str = "data", chroma_dir: str = "chroma_db"):
        """
        Initialize the document ingestor.
        
        Args:
            data_dir: Directory containing source documents
            chroma_dir: Directory for ChromaDB persistence
        """
        self.data_dir = Path(data_dir)
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(exist_ok=True)
        
        # Track processed files
        self.tracking_file = self.chroma_dir / "processed_files.json"
        self.processed_files = self._load_tracking_file()
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize text splitters
        self.handbook_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        self.course_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        self.documents: List[Document] = []
        self.new_files = []
        self.skipped_files = []
    
    def _load_tracking_file(self) -> Dict[str, Any]:
        """Load the tracking file with processed file information."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except:
                return {"files": {}, "last_updated": None}
        return {"files": {}, "last_updated": None}
    
    def _save_tracking_file(self):
        """Save the tracking file with updated information."""
        self.processed_files["last_updated"] = datetime.now().isoformat()
        with open(self.tracking_file, 'w') as f:
            json.dump(self.processed_files, f, indent=2)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file to detect changes."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _is_file_processed(self, file_path: Path) -> bool:
        """
        Check if a file has already been processed.
        Returns True if file exists in tracking and hash matches.
        """
        file_name = file_path.name
        current_hash = self._get_file_hash(file_path)
        
        if file_name in self.processed_files["files"]:
            stored_hash = self.processed_files["files"][file_name].get("hash")
            if stored_hash == current_hash:
                return True
            else:
                logger.info(f"🔄 File changed: {file_name} (hash mismatch)")
                return False
        return False
    
    def detect_document_type(self, content: str, filename: str) -> str:
        """Detect document type based on content and filename."""
        filename_lower = filename.lower()
        
        if 'handbook' in filename_lower or 'policy' in filename_lower:
            return 'handbook'
        elif 'schedule' in filename_lower or 'timetable' in filename_lower:
            return 'schedule'
        elif 'course' in filename_lower or 'outline' in filename_lower or 'memo' in filename_lower:
            return 'course_outline'
        
        content_lower = content.lower()
        if 'course code' in content_lower or 'course outline' in content_lower:
            return 'course_outline'
        elif 'schedule' in content_lower or 'time' in content_lower and 'room' in content_lower:
            return 'schedule'
        else:
            return 'handbook'
    
    def extract_metadata_from_content(self, content: str, doc_type: str) -> Dict[str, Any]:
        """Extract structured metadata from document content."""
        metadata = {}
        
        if doc_type == 'course_outline':
            # Extract course code - handles both "ACCT100" and "ACCT 100"
            pattern1 = r'\b([A-Z]{2,4}\d{3,4})\b'
            pattern2 = r'\b([A-Z]{2,4})\s+(\d{3,4})\b'
            
            matches1 = re.findall(pattern1, content)
            if matches1:
                metadata['course_code'] = matches1[0]
            else:
                matches2 = re.search(pattern2, content)
                if matches2:
                    metadata['course_code'] = f"{matches2.group(1)}{matches2.group(2)}"
            
            semester_pattern = r'(Fall|Spring|Summer)\s*\d{4}'
            semester_match = re.search(semester_pattern, content, re.IGNORECASE)
            if semester_match:
                metadata['semester'] = semester_match.group(0)
            
            title_patterns = [
                r'Course Title:?\s*([^\n]+)',
                r'Course\s+Title\s*[:\-]?\s*([^\n]+)',
                r'Title:?\s*([^\n]+)'
            ]
            for pattern in title_patterns:
                title_match = re.search(pattern, content, re.IGNORECASE)
                if title_match:
                    metadata['course_title'] = title_match.group(1).strip()
                    break
        
        elif doc_type == 'schedule':
            day_pattern = r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)'
            day_match = re.search(day_pattern, content, re.IGNORECASE)
            if day_match:
                metadata['day'] = day_match.group(0)
            
            time_pattern = r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\s*[-–]\s*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)'
            time_match = re.search(time_pattern, content)
            if time_match:
                metadata['time'] = time_match.group(0)
        
        return metadata
    
    def process_pdf(self, pdf_path: Path) -> List[Document]:
        """Process a PDF file and convert to LangChain Documents."""
        file_size_mb = pdf_path.stat().st_size / 1024 / 1024
        logger.info(f"📄 Processing: {pdf_path.name} ({file_size_mb:.2f} MB)")
        
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            logger.info(f"   Loaded {len(pages)} pages")
            
            sample_content = " ".join([doc.page_content[:500] for doc in pages[:3]])
            doc_type = self.detect_document_type(sample_content, pdf_path.name)
            logger.info(f"   Detected type: {doc_type}")
            
            splitter = self.handbook_splitter if doc_type == 'handbook' else self.course_splitter
            
            for i, doc in enumerate(pages):
                doc.metadata.update({
                    "source": pdf_path.stem,
                    "document_type": doc_type,
                    "page": i + 1,
                    "file_name": pdf_path.name,
                    "total_pages": len(pages),
                    "processed_date": datetime.now().isoformat()
                })
                
                extracted_meta = self.extract_metadata_from_content(
                    doc.page_content[:1000], 
                    doc_type
                )
                doc.metadata.update(extracted_meta)
            
            chunks = splitter.split_documents(pages)
            logger.info(f"   Created {len(chunks)} chunks")
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_path.name}: {e}")
            return []
    
    def get_new_files(self) -> List[Path]:
        """Get list of new or modified PDF files."""
        pdf_files = list(self.data_dir.glob("*.pdf"))
        new_files = []
        
        for pdf_file in pdf_files:
            if not self._is_file_processed(pdf_file):
                new_files.append(pdf_file)
            else:
                self.skipped_files.append(pdf_file.name)
        
        return new_files
    
    def ingest_all(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process all new PDF documents and store in ChromaDB.
        
        Args:
            force_reprocess: If True, reprocess all files regardless of tracking
        
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info("=" * 60)
        logger.info("🚀 STARTING INCREMENTAL INGESTION")
        logger.info("=" * 60)
        
        # Get new files
        if force_reprocess:
            logger.info("🔄 Force reprocess enabled - processing all files")
            pdf_files = list(self.data_dir.glob("*.pdf"))
            self.new_files = pdf_files
            self.skipped_files = []
        else:
            self.new_files = self.get_new_files()
        
        if not self.new_files:
            logger.info("✅ No new files to process")
            logger.info(f"📊 Total files in DB: {len(self.processed_files['files'])}")
            return {
                "processed": 0,
                "skipped": len(self.skipped_files),
                "total_in_db": len(self.processed_files['files']),
                "message": "No new files to process"
            }
        
        logger.info(f"📁 Found {len(self.new_files)} new files to process")
        logger.info(f"⏭️  Skipping {len(self.skipped_files)} already processed files")
        
        # Process each new file
        total_chunks = 0
        for pdf_file in self.new_files:
            chunks = self.process_pdf(pdf_file)
            if chunks:
                self.documents.extend(chunks)
                total_chunks += len(chunks)
                
                # Update tracking
                file_hash = self._get_file_hash(pdf_file)
                self.processed_files["files"][pdf_file.name] = {
                    "hash": file_hash,
                    "processed_date": datetime.now().isoformat(),
                    "chunks": len(chunks),
                    "type": self.detect_document_type("", pdf_file.name)
                }
        
        if not self.documents:
            logger.warning("⚠️ No documents were processed successfully")
            return {
                "processed": 0,
                "skipped": len(self.skipped_files),
                "total_in_db": len(self.processed_files['files']),
                "message": "No documents processed successfully"
            }
        
        # Store in ChromaDB
        logger.info(f"💾 Storing {len(self.documents)} new documents in ChromaDB...")
        
        try:
            # Load existing vectorstore or create new one
            vectorstore = Chroma(
                persist_directory=str(self.chroma_dir),
                embedding_function=self.embeddings,
                collection_name="lums_academic_knowledge"
            )
            
            # Add new documents
            vectorstore.add_documents(self.documents)
            vectorstore.persist()
            
            # Save tracking file
            self._save_tracking_file()
            
            # Get statistics
            collection_stats = vectorstore._collection.count()
            
            # Document type breakdown
            doc_types = {}
            for doc in self.documents:
                doc_type = doc.metadata.get("document_type", "unknown")
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            logger.info("=" * 60)
            logger.info("✅ INGESTION COMPLETE")
            logger.info("=" * 60)
            logger.info(f"📊 New files processed: {len(self.new_files)}")
            logger.info(f"📄 New chunks added: {total_chunks}")
            logger.info(f"📚 Total chunks in DB: {collection_stats}")
            logger.info(f"📂 Total files in DB: {len(self.processed_files['files'])}")
            logger.info("\n📊 New document type breakdown:")
            for doc_type, count in doc_types.items():
                logger.info(f"   - {doc_type}: {count} chunks")
            
            return {
                "processed": len(self.new_files),
                "skipped": len(self.skipped_files),
                "chunks_added": total_chunks,
                "total_chunks": collection_stats,
                "total_files": len(self.processed_files['files']),
                "doc_types": doc_types,
                "message": "Ingestion successful"
            }
            
        except Exception as e:
            logger.error(f"❌ Error storing in ChromaDB: {e}")
            return {
                "processed": len(self.new_files),
                "skipped": len(self.skipped_files),
                "error": str(e),
                "message": "Ingestion failed"
            }

def main():
    """Main ingestion function with command-line arguments."""
    import sys
    
    force = "--force" in sys.argv or "-f" in sys.argv
    
    ingestor = DocumentIngestor()
    result = ingestor.ingest_all(force_reprocess=force)
    
    print("\n" + "=" * 60)
    print("📊 INGESTION SUMMARY")
    print("=" * 60)
    for key, value in result.items():
        print(f"   {key}: {value}")
    print("=" * 60)

if __name__ == "__main__":
    main()