# app.py

import streamlit as st
from agent import get_diagnosis_chain # Import the function from your agent.py file
import time

# --- Configuration ---
st.set_page_config(
    page_title="DDx Agent",
    page_icon="🩺",
    layout="wide"
)

# --- Title and Description ---
st.title("🩺 Differential Diagnosis Assistant")
st.markdown("""
This AI powered agent assists in generating differential diagnoses based on patient symptoms. 
Enter the clinical presentation below, and the agent will retrieve information from its knowledge base to provide a list of potential conditions.
""")

# --- Sidebar Information ---
with st.sidebar:
    st.header("About This Project")
    st.markdown("""
    This application is a demonstration of a Retrieval Augmented Generation (RAG) 
    system built with LangChain and powered by local, open-source models.
    """)
    
    st.markdown("---")
    
    st.header("Technology Stack")
    st.markdown("""
    - **Framework:** LangChain
    - **LLM:** Ollama (Llama 3)
    - **Embeddings:** Sentence-Transformers
    - **Vector Store:** ChromaDB
    - **UI:** Streamlit
    """)
    
    st.markdown("---")
    
    st.warning("**Disclaimer:** This is an academic and portfolio project. It is NOT a medical device and should not be used for actual medical diagnosis or treatment.")

# --- Main Application Logic ---

# Initialize the RAG chain once and cache it for efficiency
@st.cache_resource
def load_chain():
    return get_diagnosis_chain()

chain = load_chain()

# Input text area for user to enter symptoms
symptom_input = st.text_area(
    "Enter patient symptoms, signs, and basic lab results:",
    height=150,
    placeholder="e.g., Patient presents with a sudden high fever, chills, body aches, and a severe headache."
)

# Button to trigger the diagnosis generation
if st.button("Generate Diagnosis"):
    if not symptom_input.strip():
        st.warning("Please enter symptoms before generating a diagnosis.")
    else:
        # Show a spinner while the agent is working
        with st.spinner("The agent is thinking... This may take a moment, especially on the first run."):
            try:
                # Invoke the RAG chain with the user's input
                start_time = time.time()
                response = chain.invoke(symptom_input)
                end_time = time.time()
                
                # Display the response
                st.success("Diagnosis Generated!")
                st.markdown(response)
                
                # Display the time taken
                st.info(f"Time taken: {end_time - start_time:.2f} seconds")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please ensure the Ollama application is running in the background.")