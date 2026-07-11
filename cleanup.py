import os
import shutil
from pathlib import Path

def clear_temporary_caches():
    """Wipes out all local audio chunks and cached files to free up space."""
    cache_dir = Path("data/cache")
    chunks_dir = Path("data/chunks")
    
    print("🧹 Starting project cache maintenance...")
    
    # 1. Clean out the temporary chunks folder
    if chunks_dir.exists():
        for item in chunks_dir.iterdir():
            try:
                if item.is_file():
                    os.remove(item)
            except Exception as e:
                print(f"⚠️ Could not delete chunk {item.name}: {e}")
        print("🗑️ Cleared data/chunks/ files.")
        
    # 2. Clean out the raw cache master files
    if cache_dir.exists():
        for item in cache_dir.iterdir():
            try:
                if item.is_file():
                    os.remove(item)
            except Exception as e:
                print(f"⚠️ Could not delete cache file {item.name}: {e}")
        print("🗑️ Cleared data/cache/ master audio tracks.")
        
    print("✨ Cache maintenance complete. Hard drive space recovered!")

if __name__ == "__main__":
    clear_temporary_caches()