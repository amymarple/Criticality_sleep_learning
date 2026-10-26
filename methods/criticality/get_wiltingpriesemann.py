"""Download the MR estimator of Wilting and Priesemann (2018) from the authors' repository,
https://github.com/jwilting/WiltingPriesemann2018, into this folder.

The authors' file is written for Python 2; its three print statements are rewritten as Python 3 calls, and nothing
else changes. Run once, from the repository root:

    python methods/criticality/get_wiltingpriesemann.py
"""
import hashlib
import re
import urllib.request
from pathlib import Path

COMMIT = "0c26b11ef2f90d67cbdfb8ce462b409e8cf09d62"
URL = f"https://raw.githubusercontent.com/jwilting/WiltingPriesemann2018/{COMMIT}/WiltingPriesemann.py"
SHA256 = "8aab8e74acc25c6e09114d7a31f11bb33ec6453dbf0b31096571b1085d8fa320"
OUT = Path(__file__).with_name("WiltingPriesemann.py")

src = urllib.request.urlopen(URL).read()
if hashlib.sha256(src).hexdigest() != SHA256:
    raise SystemExit(f"{URL} does not match the version used for the paper (SHA-256 {SHA256})")
text = re.sub(r"^(\s*)print (.+)$", r"\1print(\2)", src.decode("utf-8"), flags=re.M)
OUT.write_text(text, encoding="utf-8")
print("wrote", OUT)
