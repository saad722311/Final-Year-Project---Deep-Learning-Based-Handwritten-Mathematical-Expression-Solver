# src/data/tokenizer.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]


@dataclass
class TokenizerConfig:
    add_sos_eos: bool = True


class CharTokenizer:
    """
    Character-level tokenizer for LaTeX strings.
    - Good baseline: simple, debuggable, works on any LaTeX without special rules.
    """

    def __init__(self, stoi: Dict[str, int], itos: List[str], config: TokenizerConfig | None = None):
        self.stoi = stoi
        self.itos = itos
        self.config = config or TokenizerConfig()

        # Cache special token ids
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

    def decode(self, ids: List[int], remove_special: bool = True) -> str:
        chars: List[str] = []
        for i in ids:
            if remove_special and i in (self.pad_id, self.sos_id, self.eos_id):
                continue
            if 0 <= i < len(self.itos):
                tok = self.itos[i]
                # Avoid inserting the literal special tokens
                if remove_special and tok in SPECIAL_TOKENS:
                    continue
                chars.append(tok)
        return "".join(chars)

    @staticmethod
    def build_from_texts(texts: List[str], min_freq: int = 1) -> "CharTokenizer":
        # Count characters
        freq: Dict[str, int] = {}
        for t in texts:
            for ch in t:
                freq[ch] = freq.get(ch, 0) + 1

        # Build vocab list: specials first, then sorted chars for reproducibility
        chars = sorted([ch for ch, c in freq.items() if c >= min_freq])
        itos = SPECIAL_TOKENS + chars
        stoi = {tok: i for i, tok in enumerate(itos)}
        return CharTokenizer(stoi=stoi, itos=itos)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "itos": self.itos,
            "config": {"add_sos_eos": self.config.add_sos_eos},
        }
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "CharTokenizer":
        path = Path(path)
        obj = json.loads(path.read_text(encoding="utf-8"))
        itos = obj["itos"]
        stoi = {tok: i for i, tok in enumerate(itos)}
        cfg = TokenizerConfig(**obj.get("config", {}))
        return CharTokenizer(stoi=stoi, itos=itos, config=cfg)