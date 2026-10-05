"""Download the IBM Telco Customer Churn dataset into data/raw/.

We download instead of committing the data so the repo stays small, and we
check a SHA-256 hash so we know every run uses exactly the same file.
"""

import hashlib
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "telco_churn.csv"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if DEST.exists() and sha256_of(DEST) == SHA256:
        print(f"Already downloaded: {DEST}")
        return

    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, DEST)

    actual = sha256_of(DEST)
    if actual != SHA256:
        DEST.unlink()
        raise SystemExit(f"Hash mismatch: expected {SHA256}, got {actual}")
    print(f"Saved and verified: {DEST}")


if __name__ == "__main__":
    main()
