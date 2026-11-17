"""UA-DETRAC: ambil frame dari zip remote via HTTP Range, fitur ResNet18 beku."""

import io
import os
import struct
import zlib

import numpy as np
import requests
from PIL import Image
from remotezip import RemoteZip

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
URL = ("https://huggingface.co/datasets/abhineet123/ua_detrac/resolve/main/"
       "ua_detrac_test_set.zip")
N_SEQ = 6
FRAMES_PER_SEQ = 1000


def fetch_range(start, end):
    r = requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=600)
    r.raise_for_status()
    return r.content

def parse_frames(buf, infos):
    frames = []
    pos = 0
    for info in infos:
        if struct.unpack_from("<I", buf, pos)[0] != 0x04034B50:
            pos = buf.find(b"PK\x03\x04", pos)
            if pos < 0:
                break
        fnlen, extralen = struct.unpack_from("<HH", buf, pos + 26)
        start = pos + 30 + fnlen + extralen
        raw = buf[start:start + info.compress_size]
        frames.append(Image.open(io.BytesIO(zlib.decompress(raw, -15))).convert("RGB"))
        pos = start + info.compress_size
    return frames

def select_sequences(z):
    names = [n for n in z.namelist() if n.endswith(".jpg")]
    dirs = sorted({n.split("/")[0] for n in names})
    idx = np.linspace(0.08, 0.92, N_SEQ) * (len(dirs) - 1)
    return [dirs[int(round(i))] for i in idx]


def fetch_sequence(z, seq):
    infos = [i for i in z.infolist()
             if i.filename.startswith(seq + "/") and i.filename.endswith(".jpg")]
    step = max(1, len(infos) // FRAMES_PER_SEQ)
    sel = infos[::step][:FRAMES_PER_SEQ]
    start = sel[0].header_offset
    end = sel[-1].header_offset + 2 * sel[-1].compress_size + 1_000_000
    return parse_frames(fetch_range(start, end), sel)


def extract_features(frames, batch=32):
    import torch
    from torchvision import transforms
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), batch):
            xb = torch.stack([tf(f) for f in frames[i:i + batch]])
            out.append(model(xb).numpy())
    return np.vstack(out).astype(np.float64)


def build_stream(d_pca=64):
    from sklearn.decomposition import PCA

    feats = []
    with RemoteZip(URL) as z:
        for k, seq in enumerate(select_sequences(z)):
            cache = os.path.join(DATA, f"ua_detrac_feats_{k}.npy")
            if not os.path.exists(cache):
                F = extract_features(fetch_sequence(z, seq))
                np.save(cache, F)
                print(f"  {seq}: {F.shape}", flush=True)
            else:
                F = np.load(cache)
            feats.append(F)
            print(f"  seq {k} {seq} siap ({len(F)} frame)", flush=True)

    Z = np.vstack(feats)
    y = np.repeat(np.arange(N_SEQ), FRAMES_PER_SEQ)
    first = y == 0
    mu, sd = Z[first].mean(0), Z[first].std(0) + 1e-6
    Zs = (Z - mu) / sd
    pca = PCA(n_components=d_pca, random_state=0).fit(Zs[first])
    return pca.transform(Zs).astype(np.float64), y, y.copy()