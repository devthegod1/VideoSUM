Markdown
# 🎙️ VideoSUM Workspace

An advanced, locally-driven multimedia intelligence workspace designed to ingest online broadcasts (YouTube URLs) and local recording containers, execute parallelized speech-to-text transcription matrix streams, and extract synthesized executive meeting insights via Cloud LLM pipelines. Featuring a native Retrieval-Augmented Generation (RAG) system to query long-form document vector spaces directly from an interactive user interface.

---

## 🚀 Architectural Blueprint

The application architecture follows a highly optimized, decoupled data processing pipeline:

[YouTube URL / Local File]
│
▼ (pydub / ffmpeg wrapper)
[Audio Track Ingestion & 10-Min Slicing Chunks]
│
▼ (Local Whisper Engine CPU Matrix)
[Raw Text Log Transcription Node]
│
├───► [Mistral Cloud LLM Core] ───► [Executive Minutes / Summaries Report]
│
▼ (ChromaDB Vector Indexing Pipeline)
[Persistent RAG Embeddings Node] ◄───► [Interactive Chat UI Playground Input]


---

## ✨ Features

- **Decoupled Ingestion Node:** Toggle fluidly between streaming media from YouTube URLs or uploading heavy local media containers (`.mp3`, `.wav`, `.mp4`, `.mkv`, etc.).
- **Smart Slicing Core:** Auto-slices lengthy audio into uniform 10-minute structural segments using `pydub` to prevent local processing pipeline choke points.
- **Local Acoustic Core:** Runs local multi-threaded Whisper transcription matrices directly over your CPU framework.
- **Cross-Lingual Transformation Node:** Integrated translation layer to convert regional audio tracks (e.g., Hindi speech elements) cleanly into English logs.
- **Executive Synthesis Report Engine:** Leverages powerful cloud LLMs (Mistral AI) to process long-form text logs into polished, professional meeting minutes and summary records.
- **Document RAG Node:** Indexes your transcription assets into a localized vector document store (`ChromaDB`), unlocking real-time semantic query lookups.

---

## 🛠️ Installation & Setup Workspace

This project leverages `uv`, the fast Python package installer and resolver ecosystem. 

### 1. Prerequisites
Ensure you have system level `ffmpeg` installed and mapped to your global environment paths.

### 2. Clone and Navigate to Directory
```bash
git clone [https://github.com/devthegod1/VideoSUM.git](https://github.com/devthegod1/VideoSUM.git)
cd VideoSUM
3. Initialize the Virtual Environment via uv
Bash
# Create a localized virtual environment environment node
uv venv

# Activate the workspace virtual environment node
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
4. Install Project Packages
Bash
uv pip install -r requirements.txt
5. Configure Your Private API Credentials Matrix
Create a secret environment configuration file named .env inside the root repository tracking space (Note: This file is blocked by .gitignore and stays secure on your machine):

Code snippet
MISTRAL_API_KEY=your_secret_mistral_cloud_api_key_goes_here
🎮 Execution Management
To launch the high-contrast analytical workspace panel layout engine, execute the following command:

Bash
streamlit run app.py
Open your local browser gateway port (typically http://localhost:8501) to interact with the ingestion nodes.

📂 Project Directory Topography
Plaintext
VideoSUM/
├── data/                   # Audio track containers, cache maps, & database shards (Ignored)
├── src/                    # System Modular Pipeline Logic Sub-directories
│   ├── __init__.py
│   ├── audio_ingestion.py  # Audio stream downloads, track file saving, and slicing
│   ├── transcription.py    # Local Whisper CPU execution management rules
│   ├── llm_chains.py       # Mistral Cloud synthesis mapping connections
│   └── rag_pipeline.py     # ChromaDB vector indexing and retrieval functions
├── utils/                  # Secondary functional operational aids
│   ├── __init__.py
│   └── exporter.py         # Automated Plaintext and PDF document report builders
├── app.py                  # Streamlit Interface layout UI
├── requirements.txt        # Frozen software package tracking dependencies
└── README.md               # Operations manual documentation asset
🔒 Security & Data Privacy
Zero Cloud Leakage on Speech: Transcription tasks execute locally on your machine via standard CPU allocations. No third-party system holds copies of the raw audio waves.

Credential Integrity: Private API tokens are handled via dynamic runtime memory calls wrapped safely within isolated local .env storage frameworks.