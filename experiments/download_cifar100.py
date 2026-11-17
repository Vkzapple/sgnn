"""Downloader dengan resume (HTTP Range) + verifikasi ukuran."""
import os, sys, requests

URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                    "cifar-100-python.tar.gz")

total = None
with requests.get(URL, stream=True, timeout=120) as r:
    total = int(r.headers.get("content-length", 0)) or None
    if total is None:
        with open(PATH, "wb") as f:
            for ch in r.iter_content(1 << 20):
                f.write(ch)
        print("downloaded", os.path.getsize(PATH))
        sys.exit(0)
print("total", total, "MB", round(total / 1e6, 1))
for attempt in range(10):
    have = os.path.getsize(PATH) if os.path.exists(PATH) else 0
    if have >= total:
        break
    headers = {"Range": f"bytes={have}-"}
    with requests.get(URL, headers=headers, stream=True, timeout=120) as r:
        mode = "ab" if r.status_code == 206 else "wb"
        with open(PATH, mode) as f:
            for ch in r.iter_content(1 << 20):
                f.write(ch)
    print("attempt", attempt, "->", os.path.getsize(PATH), flush=True)

size = os.path.getsize(PATH)
print("final", size, "OK" if size == total else "INCOMPLETE")
sys.exit(0 if size == total else 1)
