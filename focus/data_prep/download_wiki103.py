"""
Download WikiText-103 dataset.

WikiText-103: ~100M tokens (516MB compressed)
Much larger than WikiText-2 (~2M tokens)

Usage:
    python download_wikitext103.py                  # try direct download
    python download_wikitext103.py --huggingface    # use datasets library
"""

import os
import sys


DATA_DIR = "/root/data"
EXTRACT_DIR = os.path.join(DATA_DIR, "wikitext-103")


def download_huggingface():
    """Download using datasets library — most reliable method."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        return False

    print("Downloading WikiText-103 via Hugging Face datasets...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    for split_name, hf_split in [("train", "train"), ("valid", "validation"), ("test", "test")]:
        out_path = os.path.join(EXTRACT_DIR, f"wiki.{split_name}.tokens")
        print(f"  Writing {split_name}...")
        with open(out_path, 'w', encoding='utf-8') as f:
            for example in ds[hf_split]:
                f.write(example['text'] + '\n')
        size = os.path.getsize(out_path)
        n_words = 0
        with open(out_path, 'r') as fcount:
            for line in fcount:
                n_words += len(line.split())
        print(f"    wiki.{split_name}.tokens: {size:,} bytes, ~{n_words:,} words")

    return True


def download_direct():
    """Try direct download from known URLs."""
    import urllib.request
    import zipfile
    import ssl

    urls = [
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-v1.zip",
        "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip",
    ]

    zip_path = os.path.join(DATA_DIR, "wikitext-103-v1.zip")

    for url in urls:
        print(f"Trying: {url}")
        try:
            ctx = ssl.create_default_context()
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx) as response:
                total = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                with open(zip_path, 'wb') as f:
                    while True:
                        chunk = response.read(1024 * 1024)  # 1MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded / total * 100
                            print(f"\r  Downloaded: {downloaded/(1024*1024):.0f}MB / {total/(1024*1024):.0f}MB ({pct:.0f}%)", end='')
                print()

            if os.path.getsize(zip_path) < 10000:
                print("  File too small, skipping")
                os.remove(zip_path)
                continue

            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(DATA_DIR)

            # Find extracted directory
            if not os.path.exists(EXTRACT_DIR):
                for d in os.listdir(DATA_DIR):
                    full = os.path.join(DATA_DIR, d)
                    if os.path.isdir(full) and 'wikitext-103' in d:
                        os.rename(full, EXTRACT_DIR)
                        break

            os.remove(zip_path)
            return True

        except Exception as e:
            print(f"  Failed: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)

    return False


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if already exists
    if os.path.exists(EXTRACT_DIR):
        files = os.listdir(EXTRACT_DIR)
        if any('train' in f for f in files):
            print(f"WikiText-103 already exists at {EXTRACT_DIR}")
            for f in sorted(files):
                fpath = os.path.join(EXTRACT_DIR, f)
                size = os.path.getsize(fpath)
                print(f"  {f}: {size:,} bytes")
            return

    if '--huggingface' in sys.argv or '--hf' in sys.argv:
        if download_huggingface():
            print("Done!")
            return

    # Try direct first, then huggingface
    print("Attempting direct download...")
    if download_direct():
        print("Done!")
        return

    print("\nDirect download failed. Trying datasets library...")
    if download_huggingface():
        print("Done!")
        return

    print("\nAll methods failed. Try:")
    print("  pip install datasets")
    print("  python download_wikitext103.py --huggingface")


if __name__ == "__main__":
    main()