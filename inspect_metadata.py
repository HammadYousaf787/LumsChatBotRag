# inspect_metadata.py
"""
Check what metadata is stored in ChromaDB
"""

from rag_pipeline import LUMSRAGPipeline
import json

pipeline = LUMSRAGPipeline()

# Get a sample document
sample = pipeline.vectorstore.similarity_search("What is the minimum CGPA?", k=1)

if sample:
    doc = sample[0]
    print("📄 Document Metadata:")
    print(json.dumps(doc.metadata, indent=2))
    print("\n📝 Document Content Preview:")
    print(doc.page_content[:300])
else:
    print("No documents found")