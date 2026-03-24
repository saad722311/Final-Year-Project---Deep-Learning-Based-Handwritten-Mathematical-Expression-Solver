from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict


TOKEN_RE = re.compile(
    r"""
    (\\[a-zA-Z]+)            |  # commands like \frac \sqrt
    ([{}_^])                  |  # single-char structural tokens
    (\d+)                     |  # numbers
    ([a-zA-Z])                |  # letters
    ([+\-*/=(),\[\]])         |  # common ops/punct
    (\S)                         # fallback single non-space
    """,
    re.VERBOSE,
)


def tokenize_latex(s: str) -> List[str]:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    tokens = []
    for m in TOKEN_RE.finditer(s):
        tok = next(g for g in m.groups() if g is not None)
        tokens.append(tok)
    return tokens


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad: int
    bos: int
    eos: int
    unk: int

    def encode(self, tokens: List[str], add_bos_eos: bool = True) -> List[int]:
        ids = [self.stoi.get(t, self.unk) for t in tokens]
        if add_bos_eos:
            return [self.bos] + ids + [self.eos]
        return ids

    def decode(self, ids: List[int]) -> List[str]:
        out = []
        for i in ids:
            if i == self.eos:
                break
            if i in (self.pad, self.bos):
                continue
            out.append(self.itos[i] if 0 <= i < len(self.itos) else "<UNK>")
        return out


def build_vocab(all_latex: List[str], max_vocab: int = 400) -> Vocab:
    special = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    counter = Counter()
    for s in all_latex:
        counter.update(tokenize_latex(s))

    most = [t for t, _ in counter.most_common(max(0, max_vocab - len(special)))]
    itos = special + most
    stoi = {t: i for i, t in enumerate(itos)}

    return Vocab(
        stoi=stoi,
        itos=itos,
        pad=stoi["<PAD>"],
        bos=stoi["<BOS>"],
        eos=stoi["<EOS>"],
        unk=stoi["<UNK>"],
    )