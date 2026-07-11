import os
import shutil
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.llm_chains import get_mistral_llm

# Central persistent directory for ChromaDB storage
VECTOR_DB_DIR = Path("data/vector_store")

def initialize_rag_system(transcript_text: str) -> Chroma:
    """
    Splits the meeting transcript into overlapping blocks, generates localized 
    HuggingFace sentence embeddings, and indexes them into a persistent local ChromaDB instance.
    
    Returns:
        Chroma: An active, queryable vector database instance.
    """
    print("✂️ Chunking transcript text into overlapping conversational blocks...")
    # 1000 characters is roughly 150-200 words, preserving perfect conversational context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(transcript_text)
    print(f"✅ Generated {len(chunks)} text chunks for indexing.")

    print("🧬 Loading local HuggingFace embeddings model (all-MiniLM-L6-v2) onto CPU...")
    # all-MiniLM-L6-v2 is an ultra-fast, 90MB sentence transformer ideal for consumer CPUs
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    print("🗄️ Setting up local persistent Chroma vector database...")
    # Always wipe out old data from the last meeting analysis to avoid context mixing
    if VECTOR_DB_DIR.exists():
        try:
            shutil.rmtree(VECTOR_DB_DIR)
        except Exception as e:
            print(f"⚠️ Warning cleaning directory: {e}")

    # Build and persist the vector store
    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_DIR)
    )
    
    print("✅ Vector database successfully built and stored on disk.")
    return vector_db

def query_meeting_transcript(vector_db: Chroma, user_question: str) -> str:
    """
    Performs a vector similarity search to grab the most relevant chunks from the meeting 
    transcript and uses LangChain LCEL + Mistral to answer the user's specific query.
    
    Returns:
        str: Factual, source-grounded answer text.
    """
    # 1. Fetch the top 3 closest matching transcript segments
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(user_question)
    
    # 2. Flatten documents into a clean continuous text block
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 3. Request our optimized Mistral client instance
    llm = get_mistral_llm(temperature=0.1) # Hard factual boundary lock
    
    # 4. Craft a secure prompt preventing hallucination
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert AI meeting analyst answering a question about a technical session transcript.\n"
            "Answer the question using ONLY the retrieved transcript blocks provided below. "
            "If the answer cannot be found in the context, state clearly and honestly that it was not mentioned.\n\n"
            "=== RETRIEVED TRANSCRIPT CONTEXT ===\n"
            "{context}"
        )),
        ("human", "{question}")
    ])
    
    # 5. Build our LCEL RAG execution chain
    rag_chain = prompt | llm | StrOutputParser()
    
    print(f"🔍 Performing semantic lookup for user query: '{user_question}'...")
    response = rag_chain.invoke({"context": context_text, "question": user_question})
    
    return response