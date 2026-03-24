import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Safe positional encoding that can auto-extend if seq_len > max_len.
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = self._build_pe(max_len, d_model)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def _build_pe(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def _maybe_extend(self, needed_len: int, device):
        if needed_len <= self.pe.size(1):
            return

        # extend to next power-ish (or just needed_len + buffer)
        new_len = max(needed_len, self.pe.size(1) * 2)
        new_pe = self._build_pe(new_len, self.d_model).to(device)
        # overwrite buffer
        self.pe = new_pe

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        self._maybe_extend(T, x.device)
        return x + self.pe[:, :T, :]


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for HMER.
    - Training: teacher forcing
    - Eval/Test: autoregressive generation (greedy)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        encoder_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.2,
        max_len: int = 512,
        pad_token: int = 0,
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token = pad_token

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        # CNN channels -> d_model
        self.encoder_proj = nn.Linear(encoder_dim, d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)

        print("✓ TransformerDecoder initialized")
        print(f"  - d_model={d_model}, layers={num_layers}, heads={num_heads}, dropout={dropout}")

    def _flatten_encoder(self, encoder_out):
        # encoder_out: [B, C, H, W] -> [B, HW, C] -> proj -> [B, HW, d_model]
        B, C, H, W = encoder_out.shape
        enc = encoder_out.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return self.encoder_proj(enc)

    def forward(self, encoder_out, targets, target_lengths=None, teacher_forcing_ratio=1.0):
        """
        Teacher-forcing forward pass.
        targets: [B, T] containing SOS ... EOS/PAD
        returns logits: [B, T-1, vocab]
        """
        device = encoder_out.device
        memory = self._flatten_encoder(encoder_out)  # [B, HW, d_model]

        tgt_inp = targets[:, :-1]  # [B, T-1]
        B, T = tgt_inp.shape

        tgt = self.embedding(tgt_inp) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)

        # Causal mask: [T, T]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)

        # Keep both masks boolean to avoid the "mismatched mask type" warning
        # Convert float mask to bool mask: True means "masked"
        tgt_mask_bool = tgt_mask.isinf()  # upper triangle is -inf

        # Padding mask: [B, T]
        tgt_key_padding_mask = (tgt_inp == self.pad_token)

        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask_bool,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # [B, T, d_model]

        logits = self.fc(out)  # [B, T, vocab]
        return logits, None

    @torch.no_grad()
    def generate(self, encoder_out, max_len=128, sos_token=1, eos_token=2):
        """
        Greedy autoregressive generation.
        encoder_out should be batch size 1: [1, C, H, W]
        """
        device = encoder_out.device
        memory = self._flatten_encoder(encoder_out)  # [1, HW, d_model]

        generated = [sos_token]

        for _ in range(max_len):
            tgt_inp = torch.tensor([generated], device=device)  # [1, t]
            t = tgt_inp.size(1)

            tgt = self.embedding(tgt_inp) * math.sqrt(self.d_model)
            tgt = self.pos_encoding(tgt)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(t).to(device)
            tgt_mask_bool = tgt_mask.isinf()

            tgt_key_padding_mask = (tgt_inp == self.pad_token)

            out = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask_bool,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

            logits = self.fc(out[:, -1, :])  # [1, vocab]
            next_id = int(torch.argmax(logits, dim=-1).item())
            generated.append(next_id)

            if next_id == eos_token:
                break

        return generated, None


class CNNTransformerModel(nn.Module):
    def __init__(self, encoder, decoder, device="cpu"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.to(device)

    def forward(self, images, targets, target_lengths=None, teacher_forcing_ratio=1.0):
        encoder_out = self.encoder(images)
        logits, _ = self.decoder(encoder_out, targets, target_lengths, teacher_forcing_ratio)
        return logits, None

    @torch.no_grad()
    def generate_latex(self, image, max_len=128, sos_token=1, eos_token=2, beam_width=1):
        encoder_out = self.encoder(image)
        tokens, _ = self.decoder.generate(encoder_out, max_len=max_len, sos_token=sos_token, eos_token=eos_token)
        return tokens, None

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_transformer_model(
    vocab_size,
    embed_dim,
    decoder_dim,
    encoder_dim,
    num_layers=3,
    num_heads=8,
    dropout=0.2,
    max_len=512,
    pad_token=0,
    device="cpu",
):
    """
    Factory function to build CNN + TransformerDecoder.
    We use decoder_dim as Transformer d_model (keeps Config compatible).
    """
    from models.encoder import CNNEncoder

    encoder = CNNEncoder(pretrained=True, feature_dim=encoder_dim)

    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=decoder_dim,
        encoder_dim=encoder_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_len=max_len,
        pad_token=pad_token,
    )

    model = CNNTransformerModel(encoder, decoder, device=device)

    print("\n✓ CNN-Transformer Model initialized")
    print(f"  - Device: {device}")

    return model