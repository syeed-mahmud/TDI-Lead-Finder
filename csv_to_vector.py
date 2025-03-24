import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def create_vector_db_from_csv(csv_path, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", 
                            persist_directory="./vector_db"):
    """
    Create a vector database from CSV data using HuggingFace embeddings and LangChain.
    
    Args:
        csv_path (str): Path to the CSV file
        embedding_model_name (str): Name of the HuggingFace model to use for embeddings
        persist_directory (str): Directory to save the vector database
    """
    # Read the CSV file
    print(f"Reading CSV file from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Convert all columns to text
    texts = df.astype(str).values.tolist()
    
    # Flatten the list of lists and create text chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    print("Splitting texts into chunks...")
    chunks = text_splitter.create_documents([str(text) for text in texts])
    
    # Initialize the embedding model
    print(f"Initializing embedding model: {embedding_model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Create and persist the vector database
    print("Creating vector database...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Vector database saved to {persist_directory}")
    print("Vector database creation completed successfully!")
    return vectordb

def query_vector_db(query_text, vectordb, k=3):
    """
    Query the vector database.
    
    Args:
        query_text (str): The query text
        vectordb: The vector database instance
        k (int): Number of results to return
    
    Returns:
        List of similar documents
    """
    results = vectordb.similarity_search(query_text, k=k)
    return results

if __name__ == "__main__":
    # Example usage
    csv_path = "your_data.csv"  # Replace with your CSV file path
    
    # Create vector database
    vectordb = create_vector_db_from_csv(csv_path)
    
    # Example query
    query = "Your query here"
    results = query_vector_db(query, vectordb)
    
    # Print results
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(doc.page_content) 