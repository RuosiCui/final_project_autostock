# feature_resolver.py
from __future__ import annotations
import re
from collections import defaultdict

# Optional: best‑effort fuzzy help
try:
    from rapidfuzz import process, fuzz       # pip install rapidfuzz
    _FUZZY = True
except ImportError:
    _FUZZY = False

# ------------------------------------------------------------------
# 1.  Canonical catalog  (extend freely)
# ------------------------------------------------------------------
FEATURE_CATALOG: dict[str, dict] = {
    # key          "synonyms": [phrases …]                     # no builder_fns here – AnalysisAgent already adds cols
    "rsi2":        {"synonyms": ["2 rsi", "rsi 2", "2‑day rsi"]},
    "rsi14":       {"synonyms": ["14 rsi", "rsi 14", "standard rsi"]},

    "macd":        {"synonyms": ["macd line", "mac d", "mac‑d"]},
    "macd_hist":   {"synonyms": ["macd histogram", "macd bars"]},

    "sma20":       {"synonyms": ["20 sma", "sma 20", "20‑day sma"]},
    "sma50":       {"synonyms": ["50 sma", "sma 50", "50‑day sma"]},
    "sma200":      {"synonyms": ["200 sma", "200‑day sma"]},

    "ema12":       {"synonyms": ["ema 12", "12‑period ema"]},
    "ema26":       {"synonyms": ["ema 26", "26‑period ema"]},

    "bb_width":    {"synonyms": ["bollinger width", "bb width"]},
    "bb_upper":    {"synonyms": ["bollinger upper", "bb upper", "upper bb"]},
    "bb_lower":    {"synonyms": ["bollinger lower", "bb lower", "lower bb"]},

    "atr":         {"synonyms": ["atr", "average true range"]},
    "stoch_k":     {"synonyms": ["stochastic", "stoch k", "%k"]},

    "vol_ratio":   {"synonyms": ["relative volume", "vol ratio", "volume / avg", "volume ratio"]},
}

# quick reverse‑lookup table: phrase → key
_PHRASE2KEY = {
    phrase.lower(): key
    for key, meta in FEATURE_CATALOG.items()
    for phrase in meta["synonyms"]
}

# ------------------------------------------------------------------
# 2.  Regex helpers that catch numeric variations the table can’t list exhaustively
# ------------------------------------------------------------------
_REGEX_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(rsi)\s*([0-9]{1,3})\b", re.I),        "rsi{n}"),
    (re.compile(r"\b(sma|ma)\s*([0-9]{1,3})\b", re.I),     "sma{n}"),
    (re.compile(r"\bema\s*([0-9]{1,3})\b", re.I),          "ema{n}"),
]

# ------------------------------------------------------------------
# 3.  Public API
# ------------------------------------------------------------------
def extract_features(user_sentence: str) -> list[str]:
    """
    Parse `user_sentence` and return a list of canonical feature keys
    present in FEATURE_CATALOG (or regex‑generated ones like 'sma123').

    Unknown phrases are ignored; duplicates removed; order preserved.
    """
    if not user_sentence:
        return []

    wanted: list[str] = []
    text = user_sentence.lower()

    # ① direct synonym hits
    for phrase, key in _PHRASE2KEY.items():
        if phrase in text and key not in wanted:
            wanted.append(key)

    # ② regex matches (sma 34 etc.)
    for pat, tmpl in _REGEX_PATTERNS:
        for m in pat.finditer(text):
            n = m.group(2) if len(m.groups()) > 1 else m.group(1)
            key = tmpl.format(n=n)
            if key not in wanted:
                wanted.append(key)

    # ③ fuzzy fallback
    if _FUZZY:
        # gather candidate hits whose ratio > 80
        matches = process.extract(
            text,
            _PHRASE2KEY.keys(),
            scorer=fuzz.partial_ratio,
            score_cutoff=80,
            limit=10,
        )
        for phrase, _score, _ in matches:
            key = _PHRASE2KEY[phrase]
            if key not in wanted:
                wanted.append(key)

    # final sanity: keep only columns that AnalysisAgent *will* create
    available_cols = FEATURE_CATALOG.keys() | _regex_dynamic_keys()
    wanted = [k for k in wanted if k in available_cols]

    return wanted

# helper to list dynamic keys like "sma{n}", "ema{n}", "rsi{n}"
def _regex_dynamic_keys():
    dyn = set()
    for pref in ("sma", "ema", "rsi"):
        dyn.update(f"{pref}{n}" for n in range(1, 301))
    return dyn
# ----------------------------------------------------------------------
# 4.  Tiny CLI / self‑test
# ----------------------------------------------------------------------
if __name__ == "__main__":                        # python feature_resolver.py
    samples = [
        "Please train with 2‑day RSI, Bollinger band width and the 20‑day SMA.",
        "Use mac‑d histogram + ema 12 / ema26 plus average true range",
        "Give me relative volume and stochastic",
        "rsi14, sma50 and sma200 only",
        "Could you include ma 30 and rsi 7?",
        "volume ratio, lower bb, upper bb"
    ]

    for s in samples:
        print(f"\n► Prompt:  {s}")
        print("  → features:", extract_features(s))
