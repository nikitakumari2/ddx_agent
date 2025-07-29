# 🩺 Differential Diagnosis (DDx) Agent
This project is an AI-powered agent designed to assist with medical differential diagnosis. It leverages a Retrieval-Augmented Generation (RAG) architecture to provide evidence-based suggestions from a curated medical knowledge base, running entirely on local, open-source models.

# 🚀 How It Works: The RAG Pipeline
The agent follows a systematic process to transform a user's query into a justified diagnosis. This visual pipeline demonstrates the flow of data and logic from start to finish.

graph TD
    subgraph "Phase 1: Knowledge Base Setup (Offline)"
        A[1. Medical Text Files] -->|Documents| B(2. Document Loader);
        B -->|Chunks| C(3. Embedding Model);
        C -->|Vectors| D[4. ChromaDB Vector Store];
    end
    subgraph "Phase 2: Live Diagnosis (Real-time)"
        E[5. User Input] -->|Query| F{6. RAG Agent};
        F -->|"Symptoms query"| G(7. Embedding Model);
        G -->|Query Vector| H(8. ChromaDB Search);
        H -->|"Relevant Context"| F;
        F -->|Context + Query| I(9. Local LLM - Llama 3);
        I -->|Generated Text| J[10. Final Diagnosis];
    end

# Pipeline Explained:

## Medical Text Files: The process starts with a curated collection of .txt files in the data_sources folder, each containing information about a specific disease.

## Document Loader (LangChain): The build_vector_store.py script loads these files.

## Embedding Model (Sentence-Transformers): The text is chunked and converted into numerical vectors (embeddings) by a local model (all-MiniLM-L6-v2).

## ChromaDB Vector Store: These vectors are stored in a local database, creating a searchable knowledge base.

## User Input (Streamlit UI): The user enters patient symptoms into the web application.

## RAG Agent: The core logic receives the query.

## Embedding Model: The agent embeds the user's query into a vector using the same model from step 3.

## ChromaDB Search: The agent uses this query vector to find the most semantically similar and relevant text chunks from the vector store.

## Local LLM (Ollama - Llama 3): The retrieved context chunks and the original user query are passed to the local language model with a detailed prompt.

## Final Diagnosis: The LLM generates a structured, evidence-based differential diagnosis, which is displayed to the user in the UI.

# Features
## 100% Local & Private: No data ever leaves your machine. All models run locally.

## Natural Language Input: Enter patient symptoms in plain English.

## Evidence-Based Justifications: Each suggested diagnosis is accompanied by a confidence score and a justification paragraph based only on the retrieved context from your knowledge base.

## Cost-Free: No API keys or paid services are required.

## Interactive UI: Built with Streamlit for a clean and user-friendly experience.

# 🛠️ Technology Stack
## AI Framework: LangChain

## LLM Server: Ollama

## Language Model: Llama 3 (8B)

## Embedding Model: sentence-transformers/all-MiniLM-L6-v2

## Vector Database: ChromaDB

## Web UI: Streamlit

## Data Analysis: Pandas, NLTK, Scikit-learn

# How to Run Locally
### Clone the repository:

git clone https://github.com/your-username/ddx-agent.git
cd ddx-agent

### Set up the environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

### Install and Run Ollama:

Download and install Ollama.

Launch the Ollama application. It must be running in the background.

### Pull the Llama 3 model:

ollama pull llama3:8b

### Build the Vector Store:

Add your .txt files to the data_sources/ directory.

Run the script to build the knowledge base (this will download the embedding model on the first run):

python build_vector_store.py

### Launch the Application:

streamlit run app.py

# ⚠️ Disclaimer
This is an academic and portfolio project designed to demonstrate RAG architecture with local models. It is NOT a medical device and must not be used for actual medical diagnosis, advice, or treatment. The knowledge base is limited and may not be complete or up-to-date.
