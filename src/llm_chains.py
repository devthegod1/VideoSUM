import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load variables from .env to ensure the Mistral key is available
load_dotenv()

def get_mistral_llm(temperature: float = 0.2) -> ChatMistralAI:
    """Initializes and returns the Mistral LLM instance."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("❌ Error: MISTRAL_API_KEY not found in environment or .env file.")
        
    return ChatMistralAI(
        model="mistral-large-latest",
        temperature=temperature,
        mistral_api_key=api_key
    )

def chunk_transcript_by_time_cap(text: str, chunk_size: int = 18000) -> list[str]:
    """
    Slices the transcript text into blocks equivalent to roughly 20 minutes of speech
    (18,000 characters), with a 1,500-character overlap to keep context intact.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=1500)
    return splitter.split_text(text)


# ==========================================
# 1. QUICK NARRATIVE SUMMARY PIPELINE
# ==========================================

def generate_quick_summary(transcript_text: str) -> str:
    """
    Generates a concise, fluid narrative summary of the transcript.
    Automatically splits long transcripts into 20-minute blocks if necessary.
    """
    # 20-minute text threshold cap
    if len(transcript_text) <= 18000:
        return _execute_summary_chain(transcript_text)
        
    print(f"⚠️ Transcript exceeds 20 minutes of text. Activating Map-Reduce Summary framework...")
    chunks = chunk_transcript_by_time_cap(transcript_text)
    intermediate_summaries = []
    
    # Map Step
    for i, chunk in enumerate(chunks):
        print(f"🧠 Summarizing 20-minute segment {i+1}/{len(chunks)}...")
        chunk_summary = _execute_summary_chain(chunk)
        intermediate_summaries.append(chunk_summary)
        
    # Reduce Step
    print("🤝 Consolidating segment summaries into a master executive summary...")
    combined_notes = "\n\n".join(intermediate_summaries)
    
    llm = get_mistral_llm(temperature=0.3)
    reduce_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert technical editor. You are given a series of summaries covering different parts "
            "of a long meeting recording. Synthesize these notes into a final, highly polished master narrative summary. "
            "Capture the overall purpose of the session, major topics covered, and primary takeaways. "
            "Keep the response focused, fluid, and restricted to 2-3 comprehensive paragraphs."
        )),
        ("human", "Here are the partial summaries to merge:\n\n{notes}")
    ])
    
    reduce_chain = reduce_prompt | llm | StrOutputParser()
    return reduce_chain.invoke({"notes": combined_notes})

def _execute_summary_chain(text: str) -> str:
    """Internal helper to execute a fluid narrative summary on a single text chunk."""
    llm = get_mistral_llm(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert technical writer. Synthesize the provided transcript segment into a concise, "
            "fluid narrative summary. Capture the main topics discussed and primary conclusions reached. "
            "Weave the insights into a clear, cohesive narrative of 1-2 paragraphs without using bulleted lists."
        )),
        ("human", "Here is the transcript segment to condense:\n\n{transcript}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"transcript": text})


# ==========================================
# 2. DETAILED STRUCTURED MINUTES PIPELINE
# ==========================================

def generate_meeting_minutes(transcript_text: str) -> str:
    """
    Generates structured meeting minutes (Summary, Decisions, Action Items).
    Automatically splits long transcripts into 20-minute blocks if necessary.
    """
    if len(transcript_text) <= 18000:
        return _execute_minutes_chain(transcript_text)
        
    print(f"⚠️ Transcript exceeds 20 minutes of text. Activating Map-Reduce Minutes framework...")
    chunks = chunk_transcript_by_time_cap(transcript_text)
    intermediate_minutes = []
    
    # Map Step
    for i, chunk in enumerate(chunks):
        print(f"🧠 Extracting structural minutes from 20-minute segment {i+1}/{len(chunks)}...")
        chunk_minutes = _execute_minutes_chain(chunk)
        intermediate_minutes.append(chunk_minutes)
        
    # Reduce Step
    print("🤝 Combining all segment insights into the master executive minutes...")
    combined_notes = "\n\n=== NEXT SECTION INSIGHTS ===\n\n".join(intermediate_minutes)
    
    llm = get_mistral_llm(temperature=0.1)
    reduce_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an elite corporate editor. You are given a collection of preliminary meeting minutes "
            "taken from different segments of a long meeting. Your task is to consolidate them into a single, unified, "
            "and professional master document with exactly three clear sections:\n\n"
            "## 📝 Executive Summary\n"
            "Merge the summaries into one coherent, fluid high-level overview.\n\n"
            "## 🎯 Key Decisions\n"
            "Consolidate all decisions, removing any exact duplicates.\n\n"
            "## 🏃 Action Items\n"
            "Combine all tasks into a single clean list. Maintain details and explicit Assignees. Use 'Unassigned' if none is clear.\n\n"
            "Maintain an objective, formal, corporate tone. Do not invent details."
        )),
        ("human", "Here are the segmented meeting minutes to consolidate:\n\n{notes}")
    ])
    
    reduce_chain = reduce_prompt | llm | StrOutputParser()
    return reduce_chain.invoke({"notes": combined_notes})

def _execute_minutes_chain(text: str) -> str:
    """Internal helper to execute structured minutes extraction on a single text chunk."""
    llm = get_mistral_llm(temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an elite executive secretary and technical meeting assistant. "
            "Analyze the provided transcript segment and generate a highly organized document with three sections:\n\n"
            "## 📝 Executive Summary\n"
            "Concise high-level summary of themes discussed in this segment.\n\n"
            "## 🎯 Key Decisions\n"
            "Bullet points of significant decisions made during this segment.\n\n"
            "## 🏃 Action Items\n"
            "List actionable tasks assigned during this segment with explicit Assignees. Use 'Unassigned' if none is clear.\n\n"
            "Maintain an objective, formal tone. Do not invent details."
        )),
        ("human", "Here is the transcript segment:\n\n{transcript}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"transcript": text})