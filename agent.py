from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---
DB_PATH = 'chroma_db'
PROMPT_TEMPLATE = """
You are an expert biomedical assistant. You will be given a user's symptoms and context from a medical knowledge base. 
Your task is to generate a differential diagnosis. For each potential disease, provide a 'Confidence Score' (Low, Medium, High) and a 'Justification' paragraph explaining why the symptoms match, based ONLY on the provided context. 
Do not use outside knowledge. If the context is insufficient, state that you cannot form a diagnosis based on the information provided.

CONTEXT:
{context}

QUESTION:
{question}

DIFFERENTIAL DIAGNOSIS:
"""

def get_diagnosis_chain():
    """
    Creates and returns the RAG chain for the diagnosis agent using local models.
    """
    # Initialize the components
    
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Use local Ollama model
    llm = ChatOllama(model="llama3:8b")

    # Create RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Main block for testing ---
if __name__ == '__main__':
    chain = get_diagnosis_chain()
    test_query = "Patient presents with a sudden high fever, chills, body aches, and a severe headache."
    print("--- Testing Agent ---")
    print(f"Query: {test_query}")
    response = chain.invoke(test_query)
    print("\nResponse:")
    print(response)
