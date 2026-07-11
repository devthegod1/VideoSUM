import static_ffmpeg
static_ffmpeg.add_paths()

import os
import streamlit as st
from pathlib import Path
from pydub import AudioSegment

# Import custom modular backend pipes
from src.audio_ingestion import download_youtube_audio, save_uploaded_file, chunk_audio_file
from src.transcription import transcribe_large_audio_pipeline
from src.llm_chains import generate_meeting_minutes, generate_quick_summary
from src.rag_pipeline import initialize_rag_system, query_meeting_transcript

# Page Settings - Forcing wide layout view with no sidebar
st.set_page_config(
    page_title="VideoSUM Workspace",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State Variables securely
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = ""
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "cached_wav_path" not in st.session_state:
    st.session_state.cached_wav_path = None
if "metadata" not in st.session_state:
    st.session_state.metadata = {}
if "ingest_mode" not in st.session_state:
    st.session_state.ingest_mode = "youtube"

# 🎨 EXCLUSIVE CSS THEME OVERRIDE: Injecting Custom Light Blue (#01579B) App Canvas
st.markdown(
    """
    <style>
    /* Force main workspace layout canvas into clear Custom Light Blue tone */
    .stApp {
        background-color: #01579B !important;
    }
    
    /* Enforce dark/white high contrast for generic text readouts over the blue canvas */
    .stApp p, .stApp h3, .stApp span, .stApp label, .stApp div {
        color: #FFFFFF !important;
    }
    
    /* Ensure markdown headers on the main page render cleanly in crisp white */
    .stApp h1, .stApp h2, .stApp h4, .stApp h5, .stApp h6 {
        color: #FFFFFF !important;
    }
    
    /* Keep input element container text fields readable (black text inside input blocks) */
    div[data-baseweb="input"] input, div[data-baseweb="select"] div {
        color: #111827 !important;
    }
    
    /* Hide the default Streamlit sidebar collapse button completely */
    [data-testid="stSidebarCollapse"] {
        display: none !important;
    }
    
    /* Premium analysis workspace tab bar styling selectors */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E2937;
        border-radius: 4px 4px 0px 0px;
        padding: 12px 24px;
        color: #9CA3AF !important;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #01579B !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ==========================================================
# 🏢 TOP UTILITY NAVIGATION HEADER: Clean Native Columns Layout
# ==========================================================
# Create a dark container block for the top header bar background canvas
header_box = st.container()
with header_box:
    st.markdown(
        """
        <style>
        /* Custom styled dark header layout wrapper */
        .custom-header-bar {
            background-color: #111827; 
            padding: 20px 30px; 
            border-radius: 8px; 
            margin-bottom: 25px; 
            border-bottom: 3px solid #374151;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # Render layout columns inside the dark block wrapper
    h_col1, h_col2 = st.columns([8, 2])
    
    with h_col1:
        st.markdown(
            "<h1 style='color: #FFFFFF !important; margin: 0; font-family: \"Helvetica Neue\", Arial; font-weight: 800; font-size: 32px; letter-spacing: -1px;'>🎙️ VIDEOSUM</h1>", 
            unsafe_allow_html=True
        )
        
    with h_col2:
        # Aligning the badge container to float nicely on the right side
        st.markdown(
            """
            <div style='text-align: right; margin-top: 8px;'>
                <div style='background-color: #F3F4F6 !important; padding: 6px 16px; border-radius: 20px; display: inline-block; border: 1px solid #E5E7EB;'>
                    <span style='color: #000000 !important; font-family: sans-serif; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;'>
                        Workspace Active Node
                    </span>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
# ==========================================================
# 📥 BLOCK 1: TRANSCRIPTION CENTRE (Isolated Header Node)
# ==========================================================
st.markdown(
    """
    <div style="background-color: #111827; padding: 18px; border-radius: 8px; border-left: 6px solid #FFFFFF; margin-bottom: 20px; text-align: center;">
        <h2 style="color: #FFFFFF !important; font-family: 'Helvetica Neue', Arial; font-weight: 700; margin: 0; letter-spacing: 0.5px; text-transform: uppercase; font-size: 20px;">
            🎛️ Transcription Centre
        </h2>
        <p style="color: #9CA3AF !important; font-size: 13px; margin: 4px 0 0 0;">Ingest remote broadcast streams or upload physical localized recordings below</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# ==========================================================
# 📂 BLOCK 2: BUTTON-COLOR TOGGLE INGESTION SYSTEM 
# ==========================================================
sel_col1, sel_col2 = st.columns(2)

with sel_col1:
    if st.session_state.ingest_mode == "youtube":
        if st.button("🌐 YouTube URL Stream", type="primary", use_container_width=True):
            st.session_state.ingest_mode = "youtube"
    else:
        if st.button("🌐 YouTube URL Stream", type="secondary", use_container_width=True):
            st.session_state.ingest_mode = "youtube"
            st.rerun()

with sel_col2:
    if st.session_state.ingest_mode == "file":
        if st.button("📁 Local Media Asset File", type="primary", use_container_width=True):
            st.session_state.ingest_mode = "file"
    else:
        if st.button("📁 Local Media Asset File", type="secondary", use_container_width=True):
            st.session_state.ingest_mode = "file"
            st.rerun()

st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

# Data input bounding area box
input_area_box = st.container(border=True)
with input_area_box:
    if st.session_state.ingest_mode == "youtube":
        url = st.text_input("Target Video Stream URL Route:", placeholder="Enter online stream pathway link...")
        if url:
            if st.button("📥 Load Stream Target Matrix", use_container_width=True):
                with st.spinner("Extracting audio matrix elements..."):
                    try:
                        wav_path = download_youtube_audio(url)
                        st.session_state.cached_wav_path = wav_path
                        audio = AudioSegment.from_file(wav_path, format="wav")
                        st.session_state.metadata = {
                            "duration": f"{len(audio) / 60000:.2f} mins",
                            "size": f"{os.path.getsize(wav_path) / (1024*1024):.2f} MB",
                            "rate": f"{audio.frame_rate} Hz"
                        }
                        st.success("Asset successfully integrated into the active session matrix context!")
                    except Exception as e:
                        st.error(f"Asset loading error: {e}")
                        
    else:
        uploaded_file = st.file_uploader(
            "Upload multimedia container file directly:", 
            type=["mp3", "wav", "m4a", "mp4", "mkv", "mov"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            if st.button("📥 Load File Asset Target Matrix", use_container_width=True):
                with st.spinner("Isolating audio track layers..."):
                    try:
                        wav_path = save_uploaded_file(uploaded_file)
                        st.session_state.cached_wav_path = wav_path
                        audio = AudioSegment.from_file(wav_path, format="wav")
                        st.session_state.metadata = {
                            "duration": f"{len(audio) / 60000:.2f} mins",
                            "size": f"{os.path.getsize(wav_path) / (1024*1024):.2f} MB",
                            "rate": f"{audio.frame_rate} Hz"
                        }
                        st.success("Asset successfully integrated into the active session matrix context!")
                    except Exception as e:
                        st.error(f"Asset integration error: {e}")

# Strategy processing execution controls
if st.session_state.cached_wav_path:
    st.markdown("<br>", unsafe_allow_html=True)
    strategy_box = st.container(border=True)
    with strategy_box:
        st.markdown(f"**🎯 Target File Ready for Processing:** `{Path(st.session_state.cached_wav_path).name}`")
        strat_c1, strat_c2 = st.columns(2)
        with strat_c1:
            analysis_type = st.selectbox("Intelligence Extraction Focus:", ["Full Meeting Minutes", "Quick Narrative Summary"])
        with strat_c2:
            translate_hindi = st.checkbox("Language Transformation Matrix (Translate Hindi Speech ➔ English)", value=False)
            
        if st.button("🚀 Process Intelligence & Run Synthesis", type="primary", use_container_width=True):
            try:
                with st.spinner("📦 Slicing audio track layers into safe intervals..."):
                    chunks = chunk_audio_file(st.session_state.cached_wav_path, chunk_length_ms=600000)
                
                with st.spinner("🎙️ Transcribing speech waves (Whisper Local CPU)..."):
                    transcript = transcribe_large_audio_pipeline(
                        chunks, 
                        model_size="base", 
                        translate_to_english=translate_hindi
                    )
                    st.session_state.transcript = transcript
                
                with st.spinner("🧠 Requesting executive synthesis insights (Mistral Cloud AI)..."):
                    if analysis_type == "Full Meeting Minutes":
                        result = generate_meeting_minutes(transcript)
                    else:
                        result = generate_quick_summary(transcript)
                    st.session_state.analysis_result = result
                    
                with st.spinner("🧬 Indexing text nodes into local persistent vectors (ChromaDB)..."):
                    st.session_state.vector_db = initialize_rag_system(transcript)
                    
                st.toast("Intelligence synthesis completed successfully!", icon="🎉")
                st.rerun()
                
            except Exception as pipeline_err:
                st.error(f"Pipeline error encountered: {pipeline_err}")

# ==========================================================
# 📊 SLIDING RESULT PRESENTATION LAYER (Slides down contextually)
# ==========================================================
if st.session_state.transcript:
    st.markdown("---")
    st.markdown("### 📋 Active Analysis Workspace Panels")
    
    # Telemetry Analytics Metric Display Grid
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="⏱️ Asset Audio Duration", value=st.session_state.metadata.get("duration", "N/A"))
    with m2:
        st.metric(label="💾 Buffer Disk Size", value=st.session_state.metadata.get("size", "N/A"))
    with m3:
        st.metric(label="⚡ Sampling Core Frequency", value=st.session_state.metadata.get("rate", "N/A"))
    with m4:
        st.metric(label="📊 Local RAG Index Status", value="Ready", delta="Active Node", delta_color="normal")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Executive Insights Report", "📄 Full Raw Text Log", "💬 Interactive Document RAG Node"])
    
    with tab1:
        st.markdown(
            """
            <div style='background-color:#111827; padding:15px; border-left: 6px solid #FFFFFF; border-radius: 4px; margin-bottom: 20px;'>
                <h4 style='margin:0; color:#FFFFFF !important; font-family:Helvetica;'>Generated Analytical Insights Output</h4>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Safe Markdown callout (Inherits high contrast text cleanly over light blue canvas background)
        st.markdown(st.session_state.analysis_result)
        
        st.markdown("---")
        st.markdown("##### 📥 Export Summaries Archive")
        from utils.exporter import export_to_txt, export_to_pdf
        
        col1, col2 = st.columns(2)
        with col1:
            txt_buffer = export_to_txt(st.session_state.analysis_result)
            st.download_button(
                label="📄 Export Summary as Plaintext File (.txt)",
                data=txt_buffer,
                file_name="meeting_summary.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            pdf_buffer = export_to_pdf(st.session_state.analysis_result, document_title="VideoSUM Intelligence Report")
            st.download_button(
                label="📕 Export Summary as Structured PDF (.pdf)",
                data=pdf_buffer,
                file_name="meeting_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
    with tab2:
        st.markdown("##### 🔍 Full Text Transcription Record Output")
        st.text_area("Transcript Stream Container:", st.session_state.transcript, height=350, label_visibility="collapsed")
        
        st.markdown("---")
        from utils.exporter import export_to_txt
        raw_txt_buffer = export_to_txt(st.session_state.transcript)
        st.download_button(
            label="📥 Download Full Raw Transcript Plaintext File (.txt)",
            data=raw_txt_buffer,
            file_name="raw_transcript.txt",
            mime="text/plain",
            use_container_width=True
        )
        
    with tab3:
        st.markdown("##### 💬 Query Active Document Vectors")
        st.caption("Probe document storage context for decisions, timelines, or specific action statements natively.")
        
        chat_container = st.container(height=350, border=True)
        with chat_container:
            for q, a in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"):
                    st.markdown(a)
                
        user_query = st.chat_input("Query target document vector values...")
        if user_query:
            if st.session_state.vector_db:
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(user_query)
                with st.spinner("Searching indexing logs..."):
                    answer = query_meeting_transcript(st.session_state.vector_db, user_query)
                st.session_state.chat_history.append((user_query, answer))
                st.rerun()
            else:
                st.warning("Vector database mapping indices missing.")
else:
    # Default Empty Workspace Status Layout Component Block
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            """
            <div style='background-color:#111827; padding:25px; border-radius:8px; border: 1px solid #374151; text-align:center;'>
                <h4 style='color:#FFFFFF !important; margin-top:0; font-family:Helvetica; font-weight:600;'>System Core Idle</h4>
                <p style='color:#9CA3AF !important; font-size:14px; line-height:1.5; margin:0;'>
                    Workspace environment is empty. Use the Ingestion Management blocks above to process a web broadcast link or drop a local multimedia asset file to activate analysis parameters.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )