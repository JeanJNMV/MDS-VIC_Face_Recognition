from pathlib import Path

import numpy as np
from PIL import Image


def load_ORL(path: str = "data/ORL") -> tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    X = []
    y = []

    ids = sorted(int(a.name[1:]) for a in p.glob("s*") if a.is_dir() and a.name[1:].isdigit())

    for sid in ids:
        subject_dir = p / f"s{sid}"
        for img_path in sorted(subject_dir.glob("*.pgm")):
            img = Image.open(img_path)
            X.append(np.asarray(img, dtype=np.float32))
            y.append(sid)

    X = np.stack(X)  # (N, H, W)
    y = np.array(y)  # (N,)

    return X, y
