import os
import re
from pathlib import Path
import yt_dlp
from pydub import AudioSegment

import static_ffmpeg
# This downloads/locates ffmpeg binaries and adds them to the running process path
static_ffmpeg.add_paths()

CACHE_DIR = Path("data/cache")
CHUNKS_DIR = Path("data/chunks")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>| ]', '_', name)

def post_process_audio_to_spec(file_path: str) -> str:
    """
    Forces an audio file to the exact specs Whisper loves:
    WAV format, 16,000Hz Sample Rate, Mono Channel.
    """
    print(f"🔧 Optimizing audio parameters: forcing 16kHz, Mono, WAV...")
    audio = AudioSegment.from_file(file_path)
    
    # Force 1 channel (mono) and 16000Hz frame rate
    audio = audio.set_channels(1).set_frame_rate(16000)
    
    # Replace original file or create a matching WAV path
    base, _ = os.path.splitext(file_path)
    target_path = f"{base}_processed.wav"
    
    audio.export(target_path, format="wav")
    
    # Clean up the pre-processed file if it's different
    if os.path.abspath(file_path) != os.path.abspath(target_path) and os.path.exists(file_path):
        os.remove(file_path)
        
    return os.path.abspath(target_path)

def download_youtube_audio(url: str) -> str:
    """
    Downloads the audio stream from a YouTube URL and post-processes it to 16kHz Mono WAV.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(CACHE_DIR, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    
    print(f"🔄 Extracting audio from YouTube: {url}...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        base, _ = os.path.splitext(filename)
        downloaded_wav = f"{base}.wav"
        
        if os.path.exists(downloaded_wav):
            # Apply our 16kHz Mono optimization layer
            return post_process_audio_to_spec(downloaded_wav)
        raise FileNotFoundError("YouTube audio download failed.")

def save_uploaded_file(uploaded_file) -> str:
    """
    Accepts an uploaded file, extracts its audio track, saves it as an optimized 16kHz Mono WAV.
    """
    safe_name = sanitize_filename(uploaded_file.name)
    base_name, ext = os.path.splitext(safe_name)
    temp_target_path = CACHE_DIR / safe_name
    
    with open(temp_target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        if ext.lower() in ['.mp4', '.mkv', '.mov', '.avi']:
            print(f"📹 Video upload detected. Separating channels and forcing 16kHz Mono WAV...")
        else:
            print(f"🎵 Audio upload detected. Normalizing parameters to 16kHz Mono WAV...")
            
        return post_process_audio_to_spec(str(temp_target_path))
        
    finally:
        if os.path.exists(temp_target_path) and not temp_target_path.name.endswith('_processed.wav'):
            try:
                os.remove(temp_target_path)
            except OSError:
                pass

def chunk_audio_file(file_path: str, chunk_length_ms: int = 600000) -> list[str]:
    """
    Splits a pre-optimized WAV file into smaller 10-minute optimized WAV pieces.
    """
    print(f"📦 Slicing master optimized WAV into chunks: {Path(file_path).name}")
    
    audio = AudioSegment.from_file(file_path, format="wav")
    total_length = len(audio)
    
    if total_length <= chunk_length_ms:
        print("✨ File is short enough. Skipping chunking step.")
        return [os.path.abspath(file_path)]
        
    base_name = sanitize_filename(Path(file_path).stem)
    chunk_paths = []
    
    for i, start_ms in enumerate(range(0, total_length, chunk_length_ms)):
        end_ms = min(start_ms + chunk_length_ms, total_length)
        chunk = audio[start_ms:end_ms]
        
        chunk_name = f"{base_name}_chunk_{i}.wav"
        chunk_output_path = CHUNKS_DIR / chunk_name
        
        # Keep parameters identical: 16kHz, mono
        chunk = chunk.set_channels(1).set_frame_rate(16000)
        chunk.export(chunk_output_path, format="wav")
        chunk_paths.append(os.path.abspath(str(chunk_output_path)))
        
    print(f"✅ Split successfully into {len(chunk_paths)} standard 16kHz Mono WAV chunks.")
    return chunk_paths