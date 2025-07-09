import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Global variables for models and vector store
EMBEDDING_MODEL = None
FAISS_INDEX = None
FAISS_METADATA = None
LLM_PIPELINE = None
LLM_MODEL_NAME = "distilbert/distilgpt2" # A small, fast model for demo. Consider "HuggingFaceH4/zephyr-7b-beta" or "mistralai/Mistral-7B-Instruct-v0.2" for better quality if resources allow.

def load_embedding_model(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    """Loads the sentence transformer model for embeddings."""
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        print(f"Loading embedding model: {model_name}...")
        EMBEDDING_MODEL = SentenceTransformer(model_name)
    return EMBEDDING_MODEL

def load_vector_store(vector_store_dir: str = 'vector_store'):
    """Loads the FAISS index and its associated metadata."""
    global FAISS_INDEX, FAISS_METADATA
    if FAISS_INDEX is None or FAISS_METADATA is None:
        faiss_index_path = os.path.join(vector_store_dir, 'faiss_index.bin')
        metadata_path = os.path.join(vector_store_dir, 'faiss_metadata.pkl')

        if not os.path.exists(faiss_index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Vector store files not found in {vector_store_dir}. Please run embedding_and_indexing.py first.")

        print(f"Loading FAISS index from {faiss_index_path}...")
        FAISS_INDEX = faiss.read_index(faiss_index_path)
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            FAISS_METADATA = pickle.load(f)
    return FAISS_INDEX, FAISS_METADATA

def load_llm_pipeline(model_name: str = LLM_MODEL_NAME):
    """Loads the Language Model pipeline from Hugging Face."""
    global LLM_PIPELINE
    if LLM_PIPELINE is None:
        print(f"Loading LLM: {model_name}...")
        try:
            # Check for GPU availability
            device = 0 if torch.cuda.is_available() else -1
            print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

           

            # For smaller models like distilgpt2, direct pipeline works fine
            LLM_PIPELINE = pipeline(
                "text-generation",
                model=model_name,
                max_new_tokens=256, # Adjust based on expected answer length
                temperature=0.7,
                device=device,
                do_sample=True, # Important for more creative answers with smaller models
                top_k=50,
                top_p=0.95
            )
        except Exception as e:
            print(f"Error loading LLM {model_name}. Please ensure you have sufficient resources or choose a smaller model. Error: {e}")
            print("Falling back to a dummy pipeline if LLM loading fails.")
            # Fallback for development if LLM is too heavy
            class DummyPipeline:
                def __call__(self, text, **kwargs):
                    return [{"generated_text": "Dummy response: " + text.split("Question:")[1].strip()}]
            LLM_PIPELINE = DummyPipeline()
    return LLM_PIPELINE

def retrieve_documents(question: str, k: int = 5, product_filter: str = None) -> list[dict]:
    """
    Embeds the question and performs similarity search to retrieve top-k relevant chunks.
    Optionally filters by product.
    """
    embedding_model = load_embedding_model()
    faiss_index, faiss_metadata = load_vector_store()

    question_embedding = embedding_model.encode([question], convert_to_numpy=True)
    
    # Perform search
    distances, indices = faiss_index.search(question_embedding, k * 5) # Retrieve more to allow for filtering

    retrieved_chunks_with_distance = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0 and idx < len(faiss_metadata): # Ensure valid index
            metadata = faiss_metadata[idx]
            # Add distance for potential re-ranking or display
            metadata['distance'] = distances[0][i]
            retrieved_chunks_with_distance.append(metadata)

    # Apply product filter if specified
    if product_filter:
        # Normalize filter string for comparison
        norm_product_filter = product_filter.lower().replace(' ', '')
        filtered_results = []
        for chunk_meta in retrieved_chunks_with_distance:
            norm_chunk_product = chunk_meta['product'].lower().replace(' ', '')
            # Simple substring match for flexibility (e.g., "savings" matches "savings account")
            if norm_product_filter in norm_chunk_product:
                filtered_results.append(chunk_meta)
        retrieved_chunks_with_distance = filtered_results

    # Sort by distance and take top k after filtering
    retrieved_chunks_with_distance.sort(key=lambda x: x['distance'])
    final_retrieved_chunks = retrieved_chunks_with_distance[:k]

    return final_retrieved_chunks

def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Combines retrieved chunks with a prompt and sends to the LLM to generate an answer.
    """
    llm_pipeline = load_llm_pipeline()

    if not retrieved_chunks:
        return "I don't have enough information from the complaints data to answer that question. Please try rephrasing or asking about a different topic."

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(f"Source {i+1} (Product: {chunk['product']}, Complaint ID: {chunk['complaint_id']}): {chunk['text_chunk']}")
    context = "\n\n".join(context_parts)

    prompt_template = f"""You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use only the provided context to formulate your answer. If the context doesn't contain enough information to answer the question, state that you don't have enough information.
    Context:
    {context}

    Question: {question}

    Answer:
    """
    try:
        # For text-generation pipeline, provide a single string and let it complete
        # We need to extract the generated part carefully
        response = llm_pipeline(prompt_template, num_return_sequences=1)[0]['generated_text']

        # The LLM might repeat the prompt or add boilerplate. Extract the answer part.
        answer_prefix = "Answer:"
        if answer_prefix in response:
            generated_answer = response.split(answer_prefix, 1)[1].strip()
        else:
            generated_answer = response # Fallback if prefix isn't found

        # Further trim if the model generates too much or repeats the prompt
        if question in generated_answer:
            generated_answer = generated_answer.split(question)[-1].strip()
        if context_parts[0] in generated_answer: # Avoid repeating context
             generated_answer = generated_answer.split(context_parts[0])[-1].strip()

        # Simple trimming for common LLM behaviors (e.g., continuing conversation)
        if "\nQuestion:" in generated_answer:
            generated_answer = generated_answer.split("\nQuestion:")[0].strip()
        if "\nContext:" in generated_answer:
            generated_answer = generated_answer.split("\nContext:")[0].strip()

        # Ensure it doesn't sound too confident if info is missing
        if "don't have enough information" in prompt_template and \
           "I don't have enough information" not in generated_answer and \
           len(retrieved_chunks) == 0: # If no chunks were retrieved, explicitly state
            return "I don't have enough information from the complaints data to answer that question."


        return generated_answer

    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return "An error occurred while generating the answer."

if __name__ == "__main__":
    

    print("Initializing RAG components...")
    load_embedding_model()
    load_vector_store()
    load_llm_pipeline() # This might take time to download model

    test_questions = [
        "Why are people unhappy with BNPL?",
        "What are common issues with Credit Card billing disputes?",
        "Are there any fraud complaints related to Money Transfers?",
        "What are the main problems customers face with Savings accounts?",
        "Tell me about issues with Personal Loans.",
        "What are the complaints about interest rates on credit cards?",
        "I need to know about unauthorized transactions for money transfers.",
        "What complaints are there about closing savings accounts?",
        "General complaints about Credit Cards.", # Broader query
        "Tell me something completely irrelevant to financial complaints." # Test irrelevant query
    ]

    for i, q in enumerate(test_questions):
        print(f"\n--- Question {i+1}: {q} ---")
        retrieved_docs = retrieve_documents(q, k=3) # Get top 3
        print(f"Retrieved {len(retrieved_docs)} relevant chunks.")
        # for doc in retrieved_docs:
        #     print(f"  - Product: {doc['product']}, ID: {doc['complaint_id']}, Text: {doc['text_chunk'][:100]}...") # Print first 100 chars

        answer = generate_answer(q, retrieved_docs)
        print("Generated Answer:")
        print(answer)

        print("\n--- Retrieved Sources for Verification ---")
        if retrieved_docs:
            for doc in retrieved_docs:
                print(f"  - Product: {doc['product']}, Complaint ID: {doc['complaint_id']}")
                print(f"    Chunk: {doc['text_chunk'][:200]}...") # Display first 200 chars of chunk
                print("-" * 20)
        else:
            print("No sources retrieved.")

    print("\n--- Qualitative Evaluation Table Structure (for your report) ---")
    print("| Question | Generated Answer | Retrieved Sources (Product, Complaint ID) | Quality Score (1-5) | Comments/Analysis |")
    print("|----------|------------------|-----------------------------------------|---------------------|-------------------|")
    print("| Why are people unhappy with BNPL? | [LLM Generated Answer] | BNPL, ID123; BNPL, ID456 | 4 | Good summary, relevant sources. |")
    print("| What are common issues with Credit Card billing disputes? | [LLM Generated Answer] | Credit Card, ID789; Credit Card, ID101 | 5 | Accurate and precise. |")
    print("| ... | ... | ... | ... | ... |")
    print("Remember to manually fill out this table after running and inspecting outputs.")