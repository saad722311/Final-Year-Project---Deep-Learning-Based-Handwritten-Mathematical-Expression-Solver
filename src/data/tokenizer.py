# src/data/tokenizer.py
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterable

from src.utils.latex_norm import normalize_latex_label


SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]


@dataclass
class TokenizerConfig:
    add_sos_eos: bool = True


# -----------------------------
# 1) Old baseline: CharTokenizer
# -----------------------------
class CharTokenizer:
    """
    Character-level tokenizer for LaTeX strings.
    (Kept for backward compatibility / ablations)
    """

    def __init__(self, stoi: Dict[str, int], itos: List[str], config: TokenizerConfig | None = None):
        self.stoi = stoi
        self.itos = itos
        self.config = config or TokenizerConfig()

        self.pad_id = self.stoi["<PAD>"]
        self.sos_id = self.stoi["<SOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.unk_id = self.stoi["<UNK>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        if self.config.add_sos_eos:
            ids.append(self.sos_id)

        for ch in text:
            ids.append(self.stoi.get(ch, self.unk_id))

        if self.config.add_sos_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], remove_special: bool = True, stop_at_eos: bool = False) -> str:
        chars: List[str] = []
        for i in ids:
            if stop_at_eos and i == self.eos_id:
                break
            if remove_special and i in (self.pad_id, self.sos_id, self.eos_id):
                continue
            if 0 <= i < len(self.itos):
                tok = self.itos[i]
                if remove_special and tok in SPECIAL_TOKENS:
                    continue
                chars.append(tok)
        return "".join(chars)

    def count_unk(self, text: str) -> int:
        return sum(1 for ch in text if self.stoi.get(ch, self.unk_id) == self.unk_id)

    @staticmethod
    def build_from_texts(texts: List[str], min_freq: int = 1) -> "CharTokenizer":
        freq: Dict[str, int] = {}
        for t in texts:
            for ch in t:
                freq[ch] = freq.get(ch, 0) + 1
        chars = sorted([ch for ch, c in freq.items() if c >= min_freq])
        itos = SPECIAL_TOKENS + chars
        stoi = {tok: i for i, tok in enumerate(itos)}
        return CharTokenizer(stoi=stoi, itos=itos)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {"type": "char", "itos": self.itos, "config": {"add_sos_eos": self.config.add_sos_eos}}
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "CharTokenizer":
        path = Path(path)
        obj = json.loads(path.read_text(encoding="utf-8"))
        itos = obj["itos"]
        stoi = {tok: i for i, tok in enumerate(itos)}
        cfg = TokenizerConfig(**obj.get("config", {}))
        return CharTokenizer(stoi=stoi, itos=itos, config=cfg)


# ---------------------------------
# 2) New: LaTeX-aware tokenization
# ---------------------------------
_LATEX_TOKEN_RE = re.compile(
    r"""
    (\\[a-zA-Z]+)              |  # LaTeX command like \frac \alpha \leq
    (\\.)                      |  # escaped single char like \{ \} \$ \_
    (\d)                       |  # number (group digits)
    (\{|\}|\(|\)|\[|\])        |  # brackets
    (\^|_)                     |  # script operators
    (\+|-|=|,|\.|/|:|;|!|\?)   |  # common operators/punct
    (\s+)                      |  # whitespace
    (.)                           # any other single character
    """,
    re.VERBOSE,
)


def latex_lex(s: str) -> List[str]:
    """
    Tokenize LaTeX into sensible units.
    Examples:
      "\\frac{1}{2}" -> ["\\frac","{","1","}","{","2","}"]
      "x^2+y"        -> ["x","^","2","+","y"]
      "\\alpha_1"    -> ["\\alpha","_","1"]
    """
    toks: List[str] = []
    for m in _LATEX_TOKEN_RE.finditer(s):
        tok = m.group(0)
        if tok.isspace():
            # usually CROHME labels are space-light; keeping spaces hurts EM.
            # If you want to preserve spaces, replace "continue" with: toks.append(" ")
            continue
        toks.append(tok)
    return toks


class LatexTokenizer:
    """
    LaTeX-aware tokenizer. This is what your CNN-LSTM / CNN-Transformer should use.
    """

    def __init__(self, stoi: Dict[str, int], itos: List[str], config: TokenizerConfig | None = None):
        self.stoi = stoi
        self.itos = itos
        self.config = config or TokenizerConfig()

        self.pad_id = self.stoi["<PAD>"]
        self.sos_id = self.stoi["<SOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.unk_id = self.stoi["<UNK>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        if self.config.add_sos_eos:
            ids.append(self.sos_id)

        toks = latex_lex(text)
        for t in toks:
            ids.append(self.stoi.get(t, self.unk_id))

        if self.config.add_sos_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], remove_special: bool = True, stop_at_eos: bool = False) -> str:
        toks: List[str] = []
        for i in ids:
            if stop_at_eos and i == self.eos_id:
                break
            if remove_special and i in (self.pad_id, self.sos_id, self.eos_id):
                continue
            if 0 <= i < len(self.itos):
                tok = self.itos[i]
                if remove_special and tok in SPECIAL_TOKENS:
                    continue
                toks.append(tok)
        # IMPORTANT: join without spaces
        return "".join(toks)

    def count_unk(self, text: str) -> int:
        toks = latex_lex(text)
        return sum(1 for t in toks if self.stoi.get(t, self.unk_id) == self.unk_id)

    @staticmethod
    def build_from_texts(texts: List[str], min_freq: int = 1) -> "LatexTokenizer":
        freq: Dict[str, int] = {}
        for t in texts:
            for tok in latex_lex(t):
                freq[tok] = freq.get(tok, 0) + 1

        toks = sorted([tok for tok, c in freq.items() if c >= min_freq])
        itos = SPECIAL_TOKENS + toks
        stoi = {tok: i for i, tok in enumerate(itos)}
        return LatexTokenizer(stoi=stoi, itos=itos)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {"type": "latex", "itos": self.itos, "config": {"add_sos_eos": self.config.add_sos_eos}}
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "LatexTokenizer":
        path = Path(path)
        obj = json.loads(path.read_text(encoding="utf-8"))
        itos = obj["itos"]
        stoi = {tok: i for i, tok in enumerate(itos)}
        cfg = TokenizerConfig(**obj.get("config", {}))
        return LatexTokenizer(stoi=stoi, itos=itos, config=cfg)


# ---------------------------------
# 3) Shared CSV builder (choose mode)
# ---------------------------------
def build_from_labels_csv(
    csv_path: str | Path,
    text_col: str = "label",
    min_freq: int = 1,
    normalize: bool = True,
    mode: str = "latex",  # "latex" (recommended) or "char"
):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    texts: List[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        if text_col not in reader.fieldnames:
            raise ValueError(f"Column '{text_col}' not found in {csv_path}. Found columns: {reader.fieldnames}")

        for row in reader:
            t = (row.get(text_col) or "").strip()
            if not t:
                continue
            if normalize:
                t = normalize_latex_label(t)
            if t:
                texts.append(t)

    if not texts:
        raise ValueError(f"No labels found in CSV: {csv_path}")

    mode = (mode or "latex").lower()
    if mode == "char":
        return CharTokenizer.build_from_texts(texts=texts, min_freq=min_freq)
    if mode == "latex":
        return LatexTokenizer.build_from_texts(texts=texts, min_freq=min_freq)

    raise ValueError(f"Unknown tokenizer mode: {mode} (use 'latex' or 'char')")

def load_tokenizer_auto(path: str | Path):
    """
    Load tokenizer (CharTokenizer or LatexTokenizer) automatically
    based on saved JSON metadata.
    """
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))

    tok_type = obj.get("type", "char")
    if tok_type == "latex":
        return LatexTokenizer.load(path)
    elif tok_type == "char":
        return CharTokenizer.load(path)
    else:
        raise ValueError(f"Unknown tokenizer type in {path}: {tok_type}")