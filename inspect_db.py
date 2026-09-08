"""
Inspect the ChromaDB database to verify chunks and metadata
"""

import os
import sys
from pathlib import Path

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use the newer Chroma import (fixes the deprecation warning)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def inspect_database():
    """Inspect the ChromaDB database contents."""
    
    print("=" * 70)
    print("🔍 INSPECTING LUMS RAG DATABASE")
    print("=" * 70)
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load the vectorstore
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings,
        collection_name="lums_academic_knowledge"
    )
    
    # Get all documents (without embeddings to avoid NoneType error)
    all_data = vectorstore.get(include=["documents", "metadatas"])
    
    print(f"\n📊 DATABASE STATISTICS:")
    print(f"   Total chunks: {len(all_data['ids'])}")
    print(f"   Total documents: {len(all_data['documents'])}")
    print(f"   Total metadata: {len(all_data['metadatas'])}")
    
    # Show document type breakdown
    doc_types = {}
    for metadata in all_data['metadatas']:
        doc_type = metadata.get('document_type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    print(f"\n📂 DOCUMENT TYPE BREAKDOWN:")
    for doc_type, count in sorted(doc_types.items()):
        print(f"   - {doc_type}: {count} chunks")
    
    # Show sample documents by type
    print(f"\n📄 SAMPLE DOCUMENTS:")
    
    sample_docs = {}
    for i, (doc_id, metadata, text) in enumerate(zip(
        all_data['ids'], 
        all_data['metadatas'], 
        all_data['documents']
    )):
        doc_type = metadata.get('document_type', 'unknown')
        
        if doc_type not in sample_docs and len(sample_docs) < 5:
            sample_docs[doc_type] = {
                'id': doc_id,
                'metadata': metadata,
                'text_preview': text[:300] + '...' if len(text) > 300 else text
            }
    
    for doc_type, data in sample_docs.items():
        print(f"\n--- {doc_type.upper()} ---")
        print(f"ID: {data['id']}")
        print(f"Metadata: {data['metadata']}")
        print(f"Text preview: {data['text_preview']}")
        print("-" * 50)
    
    # Show all metadata fields present
    print(f"\n🏷️ METADATA FIELDS FOUND:")
    all_fields = set()
    for metadata in all_data['metadatas']:
        all_fields.update(metadata.keys())
    for field in sorted(all_fields):
        print(f"   - {field}")
    
    # Show course codes found
    course_codes = set()
    course_titles = {}
    for metadata in all_data['metadatas']:
        if 'course_code' in metadata:
            code = metadata['course_code']
            course_codes.add(code)
            if 'course_title' in metadata:
                course_titles[code] = metadata['course_title']
    
    if course_codes:
        print(f"\n📚 COURSE CODES FOUND:")
        for code in sorted(course_codes):
            title = course_titles.get(code, '')
            if title:
                print(f"   - {code}: {title}")
            else:
                print(f"   - {code}")
    
    # Show sample schedule entries
    print(f"\n📅 SCHEDULE ENTRIES (sample):")
    count = 0
    for metadata, text in zip(all_data['metadatas'], all_data['documents']):
        if metadata.get('document_type') == 'schedule' and count < 3:
            print(f"\n--- Schedule Entry {count + 1} ---")
            print(f"Day: {metadata.get('day', 'N/A')}")
            print(f"Time: {metadata.get('time', 'N/A')}")
            print(f"Course: {metadata.get('course_code', 'N/A')}")
            print(f"Text preview: {text[:200]}...")
            count += 1
    
    # Show some handbook content examples
    print(f"\n📖 HANDBOOK CONTENT SAMPLES:")
    count = 0
    for metadata, text in zip(all_data['metadatas'], all_data['documents']):
        if metadata.get('document_type') == 'handbook' and count < 3:
            page = metadata.get('page', 'N/A')
            print(f"\n--- Handbook Page {page} ---")
            print(f"Text preview: {text[:200]}...")
            count += 1
    
    print("\n" + "=" * 70)
    print("✅ INSPECTION COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    inspect_database()