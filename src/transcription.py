import os
from pathlib import Path
import whisper

import static_ffmpeg
static_ffmpeg.add_paths()

def transcribe_audio_chunk(file_path: str, model_size: str = "base", translate_to_english: bool = False) -> str:
    """
    Transcribes (or translates) a single optimized WAV file chunk using OpenAI Whisper on the CPU.
    
    Args:
        file_path (str): Absolute path to the WAV file chunk.
        model_size (str): Whisper model tier ('tiny', 'base', 'small').
        translate_to_english (bool): If True, activates Whisper's native translation task.
        
    Returns:
        str: Clean text transcript of this specific chunk.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file chunk not found at: {file_path}")
        
    # Force CPU configuration for safety on your laptop hardware
    device = "cpu"
    model = whisper.load_model(model_size, device=device)
    
    # Configure native internal execution tasks
    options = {}
    if translate_to_english:
        options["task"] = "translate"
        
    result = model.transcribe(file_path, **options)
    return result.get("text", "").strip()

def transcribe_large_audio_pipeline(chunk_paths: list[str], model_size: str = "base", translate_to_english: bool = False) -> str:
    """
    Loops through a list of pre-sliced audio chunks, transcribes them sequentially,
    and merges them into a clean master transcript string.
    
    Returns:
        str: Fully stitched meeting transcript.
    """
    full_transcript_segments = []
    total_chunks = len(chunk_paths)
    
    print(f"\n🚀 Starting Transcription Processing Loop ({total_chunks} chunk(s) to process)...")
    
    for index, path in enumerate(chunk_paths):
        print(f"⏳ Processing chunk {index + 1}/{total_chunks}: {Path(path).name}")
        
        # Process individual slice
        chunk_text = transcribe_audio_chunk(path, model_size=model_size, translate_to_english=translate_to_english)
        
        if chunk_text:
            print(f"   ↳ Extracted: \"{chunk_text[:60]}...\"")
            full_transcript_segments.append(chunk_text)
            
    # Join everything seamlessly with single spaces
    final_text = " ".join(full_transcript_segments)
    return final_text