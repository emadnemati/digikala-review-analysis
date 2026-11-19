
import re
import unicodedata
from html import unescape

_TRANSLATION = str.maketrans({
    "\u064a": "\u06cc",  # Arabic yeh to Persian yeh
    "\u0643": "\u06a9",  # Arabic kaf to Persian kaf
    "\u200b": " ",      # Zero-width space
    "\ufeff": None,
    "\u061c": None,
    "\u200e": None,
    "\u200f": None,
})
_HTML = re.compile(r"<!--.*?-->|</?[A-Za-z][^<>]*>", re.DOTALL)
_URL = re.compile(
    r"(?:https?://|www\.)[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+",
    re.IGNORECASE | re.ASCII,
)
_REPEATED_ZWNJ = re.compile("\u200c{2,}")

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", unescape(text)).translate(_TRANSLATION)
    text = _HTML.sub(" ", text)
    text = _URL.sub(" [URL] ", text)
    text = _REPEATED_ZWNJ.sub("\u200c", text)
    return " ".join(text.split()).strip(" \u200c")

if __name__ == "__main__":
    cases = [
        ("كيفيت كالا خوب نيست", "کیفیت کالا خوب نیست"),
        ("  نمی\u200cخرم   ۱۲۳ 👩‍💻  ", "نمی\u200cخرم ۱۲۳ 👩‍💻"),
        ("خوب\u200bبود &lt;br&gt; ولی گران بود", "خوب بود ولی گران بود"),
        ("لینک https://example.com/p/123 😡", "لینک [URL] 😡"),
    ]

    for original, expected in cases:
        result = normalize_text(original)
        assert result == expected, (original, result, expected)
        print(result)

    print(f"{len(cases)} normalization checks passed.")
