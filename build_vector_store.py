import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
DATA_PATH = 'data_sources/'
DB_PATH = 'chroma_db'

# --- Main Function ---
def main():
    """
    Main function to build and persist the vector store using local models.
    """
    print("Step 1: Loading documents...")
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    if not documents:
        print("No documents found in the data_sources directory. Exiting.")
        return
    print(f"Loaded {len(documents)} documents.")

    print("\nStep 2: Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    print("\nStep 3: Creating and persisting vector store with local embeddings...")
    # Model: all-MiniLM-L6-v2 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print(f"\nVector store created and persisted at: {DB_PATH}")
    print("Process complete.")

if __name__ == '__main__':
    main()

