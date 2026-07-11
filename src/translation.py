import os
from pathlib import Path

def is_transcript_hindi(text: str) -> bool:
    """
    A lightweight heuristic check to see if text contains Hindi characters.
    Devanagari script character codes sit between Unicode range U+0900 to U+097F.
    """
    if not text:
        return False
        
    # Count Devanagari characters in the text string
    hindi_char_count = sum(1 for char in text if '\u0900' <= char <= '\u097f')
    
    # If more than 5% of the document contains native Devanagari symbols, flag it as Hindi
    total_chars = len(text)
    percentage = (hindi_char_count / total_chars) * 100 if total_chars > 0 else 0
    
    return percentage > 5.0

def handle_text_translation_fallback(text: str) -> str:
    """
    A pipeline utility block. Because we leverage Whisper's native translation flag 
    (`task="translate"`) directly inside the audio loop, text-level translation 
    is handled at the source level. 
    
    This function acts as a pass-through layer for text or a spot to integrate a 
    free text API if text-only adjustments are required later.
    """
    return text