#!/usr/bin/env python3
"""
Script to manually load course documents into the vector store.

This script ensures documents are properly loaded and indexed,
which is required for the RAG chatbot to function.

Usage:
    cd backend
    uv run python load_documents.py
"""

import os
import sys

from config import config
from rag_system import RAGSystem


def load_all_documents():
    """Load all course documents into the vector store"""
    print("=" * 70)
    print("RAG Chatbot - Document Loading Script")
    print("=" * 70)

    print("\n1. Initializing RAG system...")
    rag = RAGSystem(config)

    # Get absolute path to docs folder
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(os.path.dirname(backend_dir), "docs")

    print(f"\n2. Looking for documents in: {docs_path}")

    if not os.path.exists(docs_path):
        print(f"\n❌ ERROR: docs folder not found at {docs_path}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Backend directory: {backend_dir}")
        return False

    # List available files
    all_files = os.listdir(docs_path)
    txt_files = [f for f in all_files if f.endswith(".txt")]

    print(f"\n3. Found {len(txt_files)} text files in docs folder:")
    for f in txt_files:
        file_path = os.path.join(docs_path, f)
        size = os.path.getsize(file_path)
        print(f"   - {f} ({size:,} bytes)")

    if len(txt_files) == 0:
        print("\n❌ ERROR: No .txt files found in docs folder")
        print(f"   Files present: {all_files}")
        return False

    # Check current database state
    current_count = rag.vector_store.get_course_count()
    current_titles = rag.vector_store.get_existing_course_titles()

    print(f"\n4. Current vector store state:")
    print(f"   Courses in database: {current_count}")
    if current_titles:
        print(f"   Existing titles:")
        for title in current_titles:
            print(f"      - {title}")

    # Ask user if they want to clear existing data
    if current_count > 0:
        print(f"\n⚠️  Vector store already contains {current_count} course(s)")
        response = (
            input("   Clear existing data and reload? (yes/no): ").strip().lower()
        )
        if response in ["yes", "y"]:
            print("\n5. Clearing existing data...")
            rag.vector_store.clear_all_data()
            print("   ✅ Data cleared")
        else:
            print("\n5. Keeping existing data, will skip duplicates")
    else:
        print("\n5. Vector store is empty, loading fresh data...")

    # Load documents
    print("\n6. Loading documents...")
    try:
        courses, chunks = rag.add_course_folder(docs_path, clear_existing=False)

        print(f"\n7. Loading complete!")
        print(f"   ✅ Loaded {courses} new course(s)")
        print(f"   ✅ Created {chunks} text chunks")

    except Exception as e:
        print(f"\n❌ ERROR during loading: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Verify final state
    print(f"\n8. Verifying final state...")
    final_count = rag.vector_store.get_course_count()
    final_titles = rag.vector_store.get_existing_course_titles()

    print(f"   Total courses in database: {final_count}")
    if final_titles:
        print(f"   Course titles:")
        for title in final_titles:
            print(f"      - {title}")

    if final_count == 0:
        print("\n⚠️  WARNING: No courses loaded! Check document format.")
        print("   Documents should have:")
        print("   - Course Title: ...")
        print("   - Course Instructor: ...")
        print("   - Lesson N: ...")
        return False

    # Test search functionality
    print(f"\n9. Testing search functionality...")
    try:
        results = rag.vector_store.search("introduction")
        if results.is_empty():
            print("   ⚠️  WARNING: Search returned no results")
        else:
            print(f"   ✅ Search works! Found {len(results.documents)} results")
            print(f"   Sample: {results.documents[0][:100]}...")
    except Exception as e:
        print(f"   ⚠️  WARNING: Search test failed: {e}")

    print("\n" + "=" * 70)
    print("✅ SUCCESS: Document loading complete!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  - {final_count} courses loaded")
    print(f"  - {chunks} total chunks indexed")
    print(f"  - Vector store ready for queries")
    print(f"\nNext steps:")
    print(f"  1. Ensure .env file has ANTHROPIC_API_KEY configured")
    print(f"  2. Start the server: ./run.sh")
    print(f"  3. Test a query at http://localhost:8000")
    print()

    return True


if __name__ == "__main__":
    success = load_all_documents()
    sys.exit(0 if success else 1)
