"""Download 3 weeks of the Telecom Italia Milan data into data/raw/telecom_milan/.

"Telecommunications - SMS, Call, Internet - MI" (Telecom Italia Big Data
Challenge, Harvard Dataverse, doi:10.7910/DVN/EGZHFV), released under ODbL 1.0.
Activity (SMS, calls, internet) per grid square of Milan every 10 minutes.

Only Mon 2013-11-04 to Sun 2013-11-24 (21 files, 7.3 GB): two weeks to train on,
one week to test. The full set (62 files, 20.8 GB) isn't needed yet. Each file is
checked against the MD5 checksum the repository publishes, and skipped if it's
already there and correct, so the script can be rerun after an interruption.

The repository asks for a one-time guestbook form (name, email, purpose) before
it serves the files, so a plain download is refused. Download them in a browser
from https://doi.org/10.7910/DVN/EGZHFV (select these 21 files, fill in the form),
put the .txt files directly in data/raw/telecom_milan/, then run this script to
check them.
"""

import hashlib
import urllib.error
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "telecom_milan"
URL = "https://dataverse.harvard.edu/api/access/datafile/{}"

# (Dataverse file id, file name, published MD5)
FILES = [
    (2674282, "sms-call-internet-mi-2013-11-04.txt", "151f85ab92fc0fea449ff19cb7e50a0a"),
    (2674279, "sms-call-internet-mi-2013-11-05.txt", "f8f22dfdce95bf66d6fa9f699afbdf5e"),
    (2674283, "sms-call-internet-mi-2013-11-06.txt", "1c4e7de6a559357560927796b5c63c06"),
    (2674271, "sms-call-internet-mi-2013-11-07.txt", "dc0fac4c7fccb4c3328a77f450230c5c"),
    (2674261, "sms-call-internet-mi-2013-11-08.txt", "e2c02fde5673256145b29eb73321ca12"),
    (2674268, "sms-call-internet-mi-2013-11-09.txt", "ceee41ef748750475d57f16770b0abe0"),
    (2674259, "sms-call-internet-mi-2013-11-10.txt", "4b7a506876b06565272af93048aa54b8"),
    (2674272, "sms-call-internet-mi-2013-11-11.txt", "748eee94e1924f95d323f3df1dd6f009"),
    (2674284, "sms-call-internet-mi-2013-11-12.txt", "e70f999b213bfe38b36c956cf0d1daf4"),
    (2674257, "sms-call-internet-mi-2013-11-13.txt", "f9d8e9e15d3b0ed82792dd9fb77f1ec2"),
    (2674267, "sms-call-internet-mi-2013-11-14.txt", "7b8e81a994c7c3f0d17338e42be76944"),
    (2674281, "sms-call-internet-mi-2013-11-15.txt", "51a465a1b72ec28cfa34cd27ecd40cb2"),
    (2674256, "sms-call-internet-mi-2013-11-16.txt", "f319979979e14164667d7606a3d11c57"),
    (2674266, "sms-call-internet-mi-2013-11-17.txt", "c66a2d9873ef54812a4a17444cba7451"),
    (2674260, "sms-call-internet-mi-2013-11-18.txt", "3455490d465be13d6a107e5982672008"),
    (2674280, "sms-call-internet-mi-2013-11-19.txt", "f9d64c99a62e560e754dbe40c0510786"),
    (2674263, "sms-call-internet-mi-2013-11-20.txt", "ea4904a581e9a829cebf2fdbaf679f25"),
    (2674276, "sms-call-internet-mi-2013-11-21.txt", "d2943e3d4f31a5d8f7575796d3b47368"),
    (2674269, "sms-call-internet-mi-2013-11-22.txt", "07a02618f46af85618247104bc46e974"),
    (2674277, "sms-call-internet-mi-2013-11-23.txt", "c0f44412f4d49cd7fa0edfff31ecd887"),
    (2674258, "sms-call-internet-mi-2013-11-24.txt", "8bffcfdffdfa73a18c9bcc64bf1de2b5"),
]


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for n, (file_id, name, md5) in enumerate(FILES, 1):
        dest = RAW / name
        if dest.exists() and md5_of(dest) == md5:
            print(f"[{n}/{len(FILES)}] already downloaded: {name}")
            continue
        print(f"[{n}/{len(FILES)}] downloading {name}", flush=True)
        part = dest.with_suffix(".part")
        try:
            with urllib.request.urlopen(URL.format(file_id)) as response, part.open("wb") as out:
                for block in iter(lambda: response.read(1 << 20), b""):
                    out.write(block)
        except urllib.error.HTTPError as error:
            part.unlink(missing_ok=True)
            raise SystemExit(f"The repository refused the download ({error.code}): it needs its guestbook form "
                             f"filled in first. Download the files in a browser instead (see this script's docstring).")
        if md5_of(part) != md5:
            part.unlink()
            raise SystemExit(f"Checksum mismatch for {name}")
        part.replace(dest)
        print(f"   saved and verified ({dest.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
