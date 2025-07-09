import gradio as gr
from src.rag_pipeline import retrieve_documents, generate_answer, load_embedding_model, load_vector_store, load_llm_pipeline
import os

# --- Initialization (Run once when the app starts) ---
print("Initializing RAG components for Gradio app...")
try:
    load_embedding_model()
    load_vector_store()
    load_llm_pipeline()
    print("RAG components loaded successfully.")
    initialization_success = True
except FileNotFoundError as e:
    print(f"Error during initialization: {e}")
    print("Please ensure you have run `python src/data_preprocessing.py` and `python src/embedding_and_indexing.py` first.")
    initialization_success = False
except Exception as e:
    print(f"An unexpected error occurred during RAG component loading: {e}")
    print("The chatbot might not function correctly.")
    initialization_success = False

# --- Chatbot Logic ---
def chatbot_response(message, history, product_filter_choice):
    """
    Handles a user's message, retrieves relevant complaints, and generates an AI response.
    """
    if not initialization_success:
        yield "The RAG system is not initialized. Please check the backend setup."
        return

    # history is a list of lists: [[user_msg, bot_msg], [user_msg, bot_msg], ...]
    # We don't need to process history for RAG, only the current message.

    # Retrieve relevant documents
    try:
        retrieved_docs = retrieve_documents(message, k=5, product_filter=product_filter_choice)
    except Exception as e:
        yield f"An error occurred during document retrieval: {e}"
        return

    # Generate answer using the retrieved documents
    try:
        raw_answer = generate_answer(message, retrieved_docs)
    except Exception as e:
        yield f"An error occurred during answer generation: {e}"
        return

    # Format the final response, including sources
    formatted_answer = raw_answer

    # Append sources for trust and verification
    if retrieved_docs:
        formatted_answer += "\n\n--- Sources ---"
        for i, doc in enumerate(retrieved_docs):
            formatted_answer += f"\n{i+1}. Product: {doc['product']}, Complaint ID: {doc['complaint_id']}"
            formatted_answer += f"\n   Excerpt: \"{doc['text_chunk'][:300]}...\"" # Show first 300 chars
    else:
        formatted_answer += "\n\nNo relevant sources found in the knowledge base."

    # Gradio's ChatInterface expects an iterator for streaming, so we yield the full response
    yield formatted_answer

# --- Gradio Interface ---
# Get unique product categories from the filtered data to populate the dropdown
product_categories = []
try:
    df_products = pd.read_csv('data/filtered_complaints.csv')
    product_categories = sorted(df_products['Product'].unique().tolist())
    product_categories.insert(0, "All Products") # Add "All Products" option
except Exception as e:
    print(f"Could not load product categories for dropdown: {e}")
    product_categories = ["All Products", "Credit card", "Personal loan", "Buy Now, Pay Later (BNPL)", "Savings account", "Money transfer"]


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # CrediTrust Complaint Assistant 🤖
        Ask me anything about customer complaints across various financial products.
        I'll provide synthesized answers backed by real customer feedback.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            product_dropdown = gr.Dropdown(
                choices=product_categories,
                label="Filter by Product",
                value="All Products", # Default value
                interactive=True
            )
            # You can add more filters here, e.g., by 'Issue' if needed

        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=chatbot_response,
                chatbot=gr.Chatbot(height=400),
                textbox=gr.Textbox(placeholder="Ask a question about complaints...", container=False, scale=7),
                title="Complaint Insights Chatbot",
                examples=[
                    "Why are people unhappy with BNPL?",
                    "What are common issues with Credit Card billing disputes?",
                    "Are there any fraud complaints related to Money Transfers?",
                    "Tell me about problems with Savings accounts.",
                    "What are the main issues related to personal loan applications?",
                    "Are there any complaints about unauthorized charges on Credit Cards?"
                ],
                cache_examples=False,
                clear_btn="Clear Chat",
                submit_btn="Ask CrediTrust"
            )
            # A bit of a hack to pass the dropdown value to the chatbot_response function
            # Gradio's ChatInterface is designed for simple message/history.
            # We'll use an event listener to update the fn arguments.
            # A better way might be to encapsulate the entire RAG logic in a class
            # and pass the product_filter as an instance variable, or use a state variable.

    # A workaround to pass dropdown value to chatbot_response
    # This requires modifying the `chatbot_response` signature to accept `product_filter_choice`.
    # Gradio's ChatInterface automatically handles message and history.
    # To pass additional args, we need to wrap `chatbot_response` or use gr.State.
    # Let's adjust `chatbot_response` to accept an extra argument, and wire it.

    # The `chatbot_response` function signature has been updated.
    # Gradio automatically passes the inputs from the text box and the chat history.
    # To pass the dropdown value, we'll need to create a custom Blocks interface
    # or ensure the `gr.ChatInterface` implicitly passes it if it's within the same block.

    # For simplicity, we'll assume `product_filter_choice` is directly accessible if the
    # dropdown is an input to the event.
    # A cleaner way for a complex interface would be to use `gr.State` or `gr.Blocks` with explicit inputs.

    # Let's use `gr.Blocks` and explicitly link inputs.
    # Re-writing the main Gradio interface section for better control.

with gr.Blocks(css="#chatbot {height: 400px;}") as demo_full:
    gr.Markdown(
        """
        # CrediTrust Complaint Assistant 🤖
        Ask me anything about customer complaints across various financial products.
        I'll provide synthesized answers backed by real customer feedback.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            product_dropdown = gr.Dropdown(
                choices=product_categories,
                label="Filter by Product",
                value="All Products",
                interactive=True,
                container=True,
                min_width=200
            )
            gr.Markdown("Select a product to narrow down your search for complaints.")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=400,
                label="Conversation"
            )
            msg = gr.Textbox(
                placeholder="Ask a question about complaints...",
                container=False,
                scale=7
            )
            submit_btn = gr.Button("Ask CrediTrust")
            clear_btn = gr.ClearButton([msg, chatbot])

    # Define the chat function that will be called
    def respond(message, chat_history, product_filter):
        # The chatbot_response function now correctly accepts `product_filter_choice`
        response_generator = chatbot_response(message, chat_history, product_filter)
        for chunk in response_generator:
            chat_history.append([message, chunk]) # Append new message and current chunk of response
            yield "", chat_history # Yield empty string for textbox, updated history for chatbot

    # Link events to the respond function
    msg.submit(respond, [msg, chatbot, product_dropdown], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot, product_dropdown], [msg, chatbot])
    clear_btn.click(lambda: (None, []), None, [msg, chatbot]) # Custom clear function

    # Make the UI responsive
    gr.on(
        [msg.submit, submit_btn.click],
        lambda: gr.update(value=""),
        inputs=None,
        outputs=msg
    )


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('vector_store', exist_ok=True)

    print("\n--- Important: Initial Setup Steps ---")
    print("1. Download 'consumer_complaints.csv' and place it in the 'data/' directory.")
    print("2. Run 'python src/data_preprocessing.py' to clean and filter the data.")
    print("3. Run 'python src/embedding_and_indexing.py' to create the vector store.")
    print("4. Then, run 'python app.py' to launch the Gradio interface.")
    print("--------------------------------------")

    # Launch the Gradio application
    demo_full.launch(share=False) # Set share=True to get a public link (temporarily)