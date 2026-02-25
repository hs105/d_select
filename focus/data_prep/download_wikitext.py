"""
Download WikiText-2 dataset.
Run this once before training.

Tries multiple sources in case URLs change.
Also supports: 

pip install datasets && python download_data.py --huggingface


"""

import os
import sys
import urllib.request
import zipfile
import ssl

# Multiple mirrors / URLs to try
URLS = [
    ("Hugging Face (raw zip)",
     "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-v1.zip"),
    ("Einstein AI (original)",
     "https://einstein.ai/research/blog/assets/download/wikitext/wikitext-2-v1.zip"),
    ("S3 (legacy)",
     "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"),
]

DATA_DIR = "./data"


def download_with_redirect(url, dest):
    """Download handling redirects and SSL."""
    # Create context that handles common SSL issues
    ctx = ssl.create_default_context()
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ctx) as response:
        with open(dest, 'wb') as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)


def download_from_urls(zip_path):
    """Try each URL until one works."""
    for name, url in URLS:
        print(f"Trying {name}: {url}")
        try:
            download_with_redirect(url, zip_path)
            # Verify it's actually a zip file
            if os.path.getsize(zip_path) < 1000:
                print(f"  File too small, probably an error page. Skipping.")
                os.remove(zip_path)
                continue
            return True
        except Exception as e:
            print(f"  Failed: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            continue
    return False


def download_huggingface():
    """Download using the datasets library as fallback."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets library: pip install datasets")
        return False

    print("Downloading via Hugging Face datasets library...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")

    extract_dir = os.path.join(DATA_DIR, "wikitext-2")
    os.makedirs(extract_dir, exist_ok=True)

    for split_name, hf_split in [("train", "train"), ("valid", "validation"), ("test", "test")]:
        out_path = os.path.join(extract_dir, f"wiki.{split_name}.tokens")
        with open(out_path, 'w', encoding='utf-8') as f:
            for example in ds[hf_split]:
                f.write(example['text'] + '\n')
        size = os.path.getsize(out_path)
        print(f"  wiki.{split_name}.tokens: {size:,} bytes")

    return True


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    extract_dir = os.path.join(DATA_DIR, "wikitext-2")

    if os.path.exists(extract_dir):
        files = os.listdir(extract_dir)
        if any('train' in f for f in files):
            print(f"Dataset already exists at {extract_dir}")
            for f in sorted(files):
                fpath = os.path.join(extract_dir, f)
                size = os.path.getsize(fpath)
                print(f"  {f}: {size:,} bytes")
            return

    # Check if user wants huggingface method
    if '--huggingface' in sys.argv or '--hf' in sys.argv:
        if download_huggingface():
            print("Done!")
            return
        else:
            print("Hugging Face download failed, trying direct URLs...")

    # Try direct download
    zip_path = os.path.join(DATA_DIR, "wikitext-2-v1.zip")

    if download_from_urls(zip_path):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(DATA_DIR)

        # The zip may extract to a subdirectory with slightly different name
        if not os.path.exists(extract_dir):
            for d in os.listdir(DATA_DIR):
                full = os.path.join(DATA_DIR, d)
                if os.path.isdir(full) and d.startswith("wikitext-2"):
                    os.rename(full, extract_dir)
                    break

        if os.path.exists(extract_dir):
            print(f"Extracted to {extract_dir}")
            for f in sorted(os.listdir(extract_dir)):
                fpath = os.path.join(extract_dir, f)
                size = os.path.getsize(fpath)
                print(f"  {f}: {size:,} bytes")
            os.remove(zip_path)
            print("Done!")
            return

    # Last resort: try huggingface datasets library
    print("\nDirect download failed. Trying Hugging Face datasets library...")
    if download_huggingface():
        print("Done!")
        return

    print("\nAll download methods failed. Please download manually:")
    print("  Option 1: pip install datasets && python download_data.py --huggingface")
    print("  Option 2: Download wikitext-2-v1.zip manually and extract to ./data/wikitext-2/")
    print("  URL: https://huggingface.co/datasets/Salesforce/wikitext")


if __name__ == "__main__":
    main()