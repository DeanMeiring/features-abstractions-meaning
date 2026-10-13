"""Download 8 weeks of the Telecom Italia Milan data into data/raw/telecom_milan/.

"Telecommunications - SMS, Call, Internet - MI" (Telecom Italia Big Data
Challenge, Harvard Dataverse, doi:10.7910/DVN/EGZHFV), released under ODbL 1.0.
Activity (SMS, calls, internet) per grid square of Milan every 10 minutes.

Mon 2013-11-04 to Sun 2013-11-24 (21 files, 7.3 GB): two weeks to train on, one
week to test. Plus Mon 2013-11-25 to Sun 2013-12-01 (7 files, added 2026-10-13): a
second test week, for replicating experiment 29. Plus Mon 2013-12-02 to Sun
2013-12-29 (28 files, added 2026-10-13): four more test weeks for experiment 30.
56 files in all; the rest of the full set (62 files, to 1 Jan 2014) isn't needed. Each file is
checked against the MD5 checksum the repository publishes, and skipped if it's
already there and correct, so the script can be rerun after an interruption.

The repository asks for a one-time guestbook form (name, email, purpose) before
it serves the files, so a plain download is refused. Download them in a browser
from https://doi.org/10.7910/DVN/EGZHFV (select these 56 files, fill in the form),
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
    # A fourth week (added 2026-10-13), a second test week for replicating experiment 29.
    (2674262, "sms-call-internet-mi-2013-11-25.txt", "9865c11cbfc3c907f8bce2eaa10f448c"),
    (2674270, "sms-call-internet-mi-2013-11-26.txt", "9a878cf4c279094446e5fb8f009f99bc"),
    (2674274, "sms-call-internet-mi-2013-11-27.txt", "a7d1abf21b6b11aa1725588d16d95c1f"),
    (2674278, "sms-call-internet-mi-2013-11-28.txt", "b47f3fa389889b9c804039c75968d85d"),
    (2674264, "sms-call-internet-mi-2013-11-29.txt", "4ee896fb8afb2fb1d7569119533d75c0"),
    (2674275, "sms-call-internet-mi-2013-11-30.txt", "54ae3a2aeb373d3cbd031ba32dc8e778"),
    (2674292, "sms-call-internet-mi-2013-12-01.txt", "785eed3199d70341ad3eb427ab157297"),
    # Four more weeks (added 2026-10-13), test weeks for experiment 30 (does the profile gain hold across weeks?).
    (2674287, "sms-call-internet-mi-2013-12-02.txt", "6ad0d9bc559c22efbfe1bfe33619b393"),
    (2674294, "sms-call-internet-mi-2013-12-03.txt", "a253615038abf0e191a1d3bb03d26aa2"),
    (2674307, "sms-call-internet-mi-2013-12-04.txt", "e4abdf54f50618a0d8fcf2db1cf18817"),
    (2674291, "sms-call-internet-mi-2013-12-05.txt", "42004492d87a1b76a91ad2e6b2e89f52"),
    (2674295, "sms-call-internet-mi-2013-12-06.txt", "6a236e67403ced6fa303f603dd0b80e6"),
    (2674301, "sms-call-internet-mi-2013-12-07.txt", "07e1b99c3cea4adf0ce07028ebbff601"),
    (2674311, "sms-call-internet-mi-2013-12-08.txt", "aa44fac0820e01a5c0268df7f4547cb0"),
    (2674313, "sms-call-internet-mi-2013-12-09.txt", "dc35aa2f908f60e6fc45663379b57dc8"),
    (2674309, "sms-call-internet-mi-2013-12-10.txt", "89f524e4039f111c2b0eec8667b9c33b"),
    (2674314, "sms-call-internet-mi-2013-12-11.txt", "cd7581d7d586c4741866976c5946481b"),
    (2674310, "sms-call-internet-mi-2013-12-12.txt", "79adf11aa1acb2863a44cfbc601a6ba6"),
    (2674315, "sms-call-internet-mi-2013-12-13.txt", "07d9d9b536ad330841c60659e39f09c0"),
    (2674293, "sms-call-internet-mi-2013-12-14.txt", "ad7ab3089c14b110786d86f8a9be4c91"),
    (2674306, "sms-call-internet-mi-2013-12-15.txt", "c227bcd0c387de3d88c3b52fff1512ca"),
    (2674289, "sms-call-internet-mi-2013-12-16.txt", "ccd799e2f437f7229fbea070f22c7ea3"),
    (2674316, "sms-call-internet-mi-2013-12-17.txt", "0c4d4cafc22cb209e2acb88782438482"),
    (2674312, "sms-call-internet-mi-2013-12-18.txt", "cf67bd1a84432cc6b4800488d2c5d166"),
    (2674286, "sms-call-internet-mi-2013-12-19.txt", "4dc65086f76e8ec6eb08f9dee4f60586"),
    (2674288, "sms-call-internet-mi-2013-12-20.txt", "4b8b24622a83d0a0944155a429411cab"),
    (2674303, "sms-call-internet-mi-2013-12-21.txt", "c3acc3d3cf7f946f32b283bfb823af8a"),
    (2674308, "sms-call-internet-mi-2013-12-22.txt", "355c8c198af7e372ba9e3d7b7f01a3b5"),
    (2674290, "sms-call-internet-mi-2013-12-23.txt", "1dd238f0b3111d75c9bae8654617db18"),
    (2674305, "sms-call-internet-mi-2013-12-24.txt", "6ea08442917409774e451d03ebd98332"),
    (2674304, "sms-call-internet-mi-2013-12-25.txt", "9ad51e1d206b52a54e3d6f95c31e1170"),
    (2674298, "sms-call-internet-mi-2013-12-26.txt", "ccca2ca9e50405eca22c508d4df6a57a"),
    (2674297, "sms-call-internet-mi-2013-12-27.txt", "479e93d8b6c442f0dc44085c9984ad30"),
    (2674296, "sms-call-internet-mi-2013-12-28.txt", "cd02c07e7c956b0abaee6495cf6fe830"),
    (2674317, "sms-call-internet-mi-2013-12-29.txt", "4d0cd54e39373db049374818174d67b9"),
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
