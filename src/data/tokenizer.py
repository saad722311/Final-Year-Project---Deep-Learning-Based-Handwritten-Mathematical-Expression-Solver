# src/data/tokenizer.py
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from src.utils.latex_norm import normalize_latex_label


SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]


@dataclass
class TokenizerConfig:
    add_sos_eos: bool = True


class CharTokenizer:
    """
    Character-level tokenizer for LaTeX strings.

    Notes:
    - encode() includes <SOS> and <EOS> if config.add_sos_eos=True.
    - decode() can optionally stop at EOS (VERY useful for exact-match evaluation).
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

    def decode(
        self,
        ids: List[int],
        remove_special: bool = True,
        stop_at_eos: bool = False,
    ) -> str:
        """
        Args:
            ids: list of token ids
            remove_special: remove PAD/SOS/EOS from output text
            stop_at_eos: if True, stop decoding when EOS is encountered (recommended for EM)

        Returns:
            decoded string
        """
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
        """How many UNKs would appear if we encode this text."""
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

    @staticmethod
    def build_from_labels_csv(
        csv_path: str | Path,
        text_col: str = "label",
        min_freq: int = 1,
        normalize: bool = True,
    ) -> "CharTokenizer":
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        texts: List[str] = []
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {csv_path}")
            if text_col not in reader.fieldnames:
                raise ValueError(
                    f"Column '{text_col}' not found in {csv_path}. Found columns: {reader.fieldnames}"
                )

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

        return CharTokenizer.build_from_texts(texts=texts, min_freq=min_freq)

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