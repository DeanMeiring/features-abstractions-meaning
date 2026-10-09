"""Pack the telecom setup files (not in git, since data/ is ignored) into setup/telecom_setup.zip.

So a second computer can run the telecom experiments (20-27) without downloading the raw data
or retraining a dictionary (retraining gives a different dictionary; copying keeps it the same).
Unpack from the repo root:   python scripts/make_setup_zip.py --unpack
"""

import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "setup" / "telecom_setup.zip"
FILES = [
    "data/telecom_hourly.npz",                               # hourly SMS / calls / internet per square (ODbL 1.0)
    "data/models/telecom_words.pt",                          # experiment 20's set words
    "data/models/telecom_correction_words_exp22b.pt",        # experiment 22b's healthy dictionary
    "data/models/telecom_correction_words_exp22b.gate.json", # its health gate (experiment 23 checks it)
    "data/models/telecom_predictive_words.pt",               # experiment 25's dictionary (used by 26, 26b, 27)
]
NOTE = """Telecom setup files for features-abstractions-meaning.

data/telecom_hourly.npz is derived from the Telecom Italia Big Data Challenge, Milan
(Harvard Dataverse, doi:10.7910/DVN/EGZHFV), released under the Open Database License
(ODbL 1.0, https://opendatacommons.org/licenses/odbl/1-0/). This derived file is shared
under the same licence. The .pt files are this project's own trained dictionaries.

SHA-256 of each file:
"""


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if "--unpack" in sys.argv:
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(ROOT)
        expected = dict(line.split("  ")[::-1] for line in z.read("SETUP_NOTE.txt").decode().splitlines() if "  data/" in line)
    (ROOT / "SETUP_NOTE.txt").unlink()
    for name, digest in expected.items():
        print(f"{name}: {'ok' if sha256(ROOT / name) == digest else 'HASH MISMATCH'}")
else:
    ZIP.parent.mkdir(exist_ok=True)
    note = NOTE + "".join(f"{sha256(ROOT / f)}  {f}\n" for f in FILES)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("SETUP_NOTE.txt", note)
        for f in FILES:
            z.write(ROOT / f, f)
    print(f"Wrote {ZIP.relative_to(ROOT)} ({ZIP.stat().st_size / 1e6:.1f} MB)")
