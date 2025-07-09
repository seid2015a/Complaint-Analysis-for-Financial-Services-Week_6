import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import os
import pickle
from tqdm import tqdm

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Splits a given text into chunks using LangChain's RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # Use character length for simplicity
        add_start_index=True # Keep track of original position
    )
    chunks = [doc.page_content for doc in text_splitter.create_documents([text])]
    return chunks

def create_and_index_vector_store(
    data_path: str,
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    vector_store_dir: str = 'vector_store'
):
    """
    Loads cleaned complaint data, chunks narratives, generates embeddings,
    and creates a FAISS vector store with metadata.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    if not os.path.exists(vector_store_dir):
        os.makedirs(vector_store_dir)

    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    all_chunks = []
    all_chunk_metadata = []

    print("Chunking and collecting narratives...")
    # Using tqdm for a progress bar
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing complaints"):
        narrative = row['narrative']
        complaint_id = row['Complaint ID']
        product = row['Product']
        issue = row['Issue']

        chunks = chunk_text(narrative, chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_chunk_metadata.append({
                'complaint_id': complaint_id,
                'product': product,
                'issue': issue,
                'chunk_id': f"{complaint_id}-{i}", # Unique ID for each chunk
                'text_chunk': chunk # Store the chunk text itself
            })

    print(f"Generated {len(all_chunks)} chunks.")
    if not all_chunks:
        print("No chunks generated. Please check your data and chunking parameters.")
        return

    print("Generating embeddings for all chunks...")
    # Process embeddings in batches to save memory/time
    batch_size = 32
    embeddings = []
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding chunks"):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_embeddings = model.encode(batch_chunks, convert_to_numpy=True)
        embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)

    print(f"Embeddings shape: {embeddings.shape}")

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # L2 distance for similarity search
    index.add(embeddings)

    # Save the FAISS index and metadata
    faiss_index_path = os.path.join(vector_store_dir, 'faiss_index.bin')
    metadata_path = os.path.join(vector_store_dir, 'faiss_metadata.pkl')

    faiss.write_index(index, faiss_index_path)
    with open(metadata_path, 'wb') as f:
        pickle.dump(all_chunk_metadata, f)

    print(f"FAISS index saved to {faiss_index_path}")
    print(f"Metadata saved to {metadata_path}")
    print("Vector store creation complete.")

if __name__ == "__main__":
   
    processed_data_path = '../data/filtered_complaints.csv'
    # Define chunking parameters
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50
    # Embedding model choice
    EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
    VECTOR_STORE_DIRECTORY = 'vector_store'

    create_and_index_vector_store(
        data_path=processed_data_path,
        model_name=EMBEDDING_MODEL_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        vector_store_dir=VECTOR_STORE_DIRECTORY
    )