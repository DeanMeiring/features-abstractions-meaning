"""Download the project's public datasets into data/raw/.

We download instead of committing the data so the repo stays small, and we
check a SHA-256 hash so we know every run uses exactly the same files.

Datasets:
  - IBM Telco Customer Churn (7,043 customers)
  - Fashion-MNIST (70,000 28x28 clothing images, 10 classes)
"""

import hashlib
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
FASHION = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"

# (url, where to save it, expected SHA-256)
FILES = [
    (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
        "master/data/Telco-Customer-Churn.csv",
        RAW / "telco_churn.csv",
        "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91",
    ),
    (
        FASHION + "train-images-idx3-ubyte.gz",
        RAW / "fashion_mnist" / "train-images-idx3-ubyte.gz",
        "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    ),
    (
        FASHION + "train-labels-idx1-ubyte.gz",
        RAW / "fashion_mnist" / "train-labels-idx1-ubyte.gz",
        "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    ),
    (
        FASHION + "t10k-images-idx3-ubyte.gz",
        RAW / "fashion_mnist" / "t10k-images-idx3-ubyte.gz",
        "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    ),
    (
        FASHION + "t10k-labels-idx1-ubyte.gz",
        RAW / "fashion_mnist" / "t10k-labels-idx1-ubyte.gz",
        "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
    ),
]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, dest: Path, sha256: str) -> None:
    if dest.exists() and sha256_of(dest) == sha256:
        print(f"Already downloaded: {dest.name}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)

    actual = sha256_of(dest)
    if actual != sha256:
        dest.unlink()
        raise SystemExit(f"Hash mismatch for {dest.name}: expected {sha256}, got {actual}")
    print(f"Saved and verified: {dest.name}")


def main() -> None:
    for url, dest, sha256 in FILES:
        download(url, dest, sha256)


if __name__ == "__main__":
    main()
