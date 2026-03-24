"""
LSTM Decoder for HMER
=====================
Implements the decoder that generates LaTeX token sequences from
the encoded image features using attention mechanism.

IMPORTANT FIX:
- Training is now NEXT-TOKEN prediction.
- We feed captions[:, :-1] and predict captions[:, 1:].
This prevents the model from learning the trivial "copy token" behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMDecoder(nn.Module):
    """
    LSTM-based decoder with attention for generating LaTeX sequences.
    """

    def __init__(self, vocab_size, embed_dim, decoder_dim, encoder_dim,
                 attention, dropout=0.5):
        super(LSTMDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.attention = attention

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.fc = nn.Linear(decoder_dim + encoder_dim, vocab_size)

        self.init_weights()

        print(f"✓ LSTM Decoder initialized")
        print(f"  - Vocabulary size: {vocab_size}")
        print(f"  - Embedding dim: {embed_dim}")
        print(f"  - Decoder dim: {decoder_dim}")
        print(f"  - Dropout: {dropout}")

    def init_weights(self):
        """Initialize embedding and linear layer weights."""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Initialize LSTM hidden state and cell state from encoder output.
        """
        mean_encoder_out = encoder_out.mean(dim=[2, 3])  # [B, encoder_dim]
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths, teacher_forcing_ratio=1.0):
        """
        Forward pass for training (with optional scheduled sampling).

        IMPORTANT:
        - Input tokens  = captions[:, :-1]
        - Target tokens = captions[:, 1:]
        - Predictions are returned with shape [B, max_len-1, vocab_size]
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(1)
        H, W = encoder_out.size(2), encoder_out.size(3)
        num_pixels = H * W

        # Flatten encoder output: [B, enc_dim, H, W] -> [B, num_pixels, enc_dim]
        encoder_out_flat = encoder_out.permute(0, 2, 3, 1).reshape(batch_size, num_pixels, encoder_dim)

        # Shift captions
        # Example: captions = [SOS, a, +, b, EOS, PAD, PAD]
        # inputs  = [SOS, a, +, b, EOS, PAD]
        # targets = [a,   +, b, EOS, PAD, PAD]
        input_tokens = encoded_captions[:, :-1]
        target_tokens = encoded_captions[:, 1:]

        max_len = input_tokens.size(1)  # now max_len-1 effectively

        # Initialize LSTM states
        h, c = self.init_hidden_state(encoder_out)

        predictions = torch.zeros(batch_size, max_len, self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max_len, num_pixels).to(encoder_out.device)

        embeddings = self.embedding(input_tokens)  # [B, max_len, embed_dim]

        # For scheduled sampling
        prev_token = input_tokens[:, 0]  # should be SOS

        for t in range(max_len):
            context, alpha = self.attention(encoder_out_flat, h)
            alphas[:, t, :] = alpha

            use_teacher_forcing = (torch.rand(1).item() < teacher_forcing_ratio)

            if use_teacher_forcing or t == 0:
                current_embedding = embeddings[:, t, :]
            else:
                current_embedding = self.embedding(prev_token)

            lstm_input = torch.cat([current_embedding, context], dim=1)
            lstm_input = self.dropout(lstm_input)

            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fc(torch.cat([h, context], dim=1))  # [B, vocab]
            predictions[:, t, :] = output

            if not use_teacher_forcing and t < max_len - 1:
                prev_token = output.argmax(dim=1)

        return predictions, alphas

    def generate(self, encoder_out, max_len=50, sos_token=1, eos_token=2, temperature=1.0, pad_token=0):
        """
        Generate LaTeX sequence autoregressively (inference).
        """
        batch_size = encoder_out.size(0)
        assert batch_size == 1, "Batch size must be 1 for generation"

        encoder_dim = encoder_out.size(1)
        H, W = encoder_out.size(2), encoder_out.size(3)
        num_pixels = H * W

        encoder_out_flat = encoder_out.permute(0, 2, 3, 1).reshape(batch_size, num_pixels, encoder_dim)

        h, c = self.init_hidden_state(encoder_out)

        current_token = torch.tensor([sos_token], device=encoder_out.device)

        generated_tokens = []
        alphas_list = []

        for t in range(max_len):
            embedding = self.embedding(current_token)

            context, alpha = self.attention(encoder_out_flat, h)
            alphas_list.append(alpha.squeeze(0).cpu())

            lstm_input = torch.cat([embedding, context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fc(torch.cat([h, context], dim=1))

            # Mask special tokens that shouldn't be generated
            output[:, sos_token] = -float('inf')
            output[:, pad_token] = -float('inf')

            if temperature == 1.0:
                predicted_token = output.argmax(dim=1)
            else:
                probs = torch.softmax(output / temperature, dim=1)
                predicted_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            current_token = predicted_token

            token_id = predicted_token.item()
            generated_tokens.append(token_id)

            if token_id == eos_token:
                break

        alphas = torch.stack(alphas_list) if alphas_list else None
        return generated_tokens, alphas

    def beam_search(self, encoder_out, beam_width=5, max_len=50, sos_token=1, eos_token=2, pad_token=0):
        """
        Generate LaTeX using beam search.
        """
        batch_size = encoder_out.size(0)
        assert batch_size == 1, "Batch size must be 1 for beam search"

        encoder_dim = encoder_out.size(1)
        H, W = encoder_out.size(2), encoder_out.size(3)
        num_pixels = H * W

        encoder_out_flat = encoder_out.permute(0, 2, 3, 1).reshape(batch_size, num_pixels, encoder_dim)

        h_init, c_init = self.init_hidden_state(encoder_out)

        beams = [([sos_token], 0.0, h_init, c_init)]
        completed_beams = []

        for t in range(max_len):
            candidates = []

            for seq, score, h, c in beams:
                if seq[-1] == eos_token:
                    completed_beams.append((seq, score))
                    continue

                current_token = torch.tensor([seq[-1]], device=encoder_out.device)
                embedding = self.embedding(current_token)

                context, _ = self.attention(encoder_out_flat, h)

                lstm_input = torch.cat([embedding, context], dim=1)
                h_new, c_new = self.lstm_cell(lstm_input, (h, c))

                output = self.fc(torch.cat([h_new, context], dim=1))

                output[:, sos_token] = -float('inf')
                output[:, pad_token] = -float('inf')

                log_probs = torch.log_softmax(output, dim=1).squeeze(0)

                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    token_id = topk_indices[i].item()
                    token_log_prob = topk_log_probs[i].item()

                    new_seq = seq + [token_id]
                    new_score = score + token_log_prob

                    candidates.append((new_seq, new_score, h_new, c_new))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

            if len(completed_beams) >= beam_width:
                break

        for seq, score, _, _ in beams:
            if seq[-1] != eos_token:
                seq = seq + [eos_token]
            completed_beams.append((seq, score))

        if completed_beams:
            completed_beams.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
            best_sequence, best_score = completed_beams[0]
            best_sequence = [t for t in best_sequence if t not in [sos_token, eos_token]]
            return best_sequence, best_score

        return beams[0][0][1:], beams[0][1]


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 Testing LSTM Decoder (SHIFTED TRAINING)")
    print("="*70 + "\n")

    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from attention import BahdanauAttention

    batch_size = 4
    vocab_size = 346
    embed_dim = 256
    decoder_dim = 512
    encoder_dim = 512
    attention_dim = 512
    max_len = 20
    H, W = 8, 8

    attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)

    decoder = LSTMDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        encoder_dim=encoder_dim,
        attention=attention,
        dropout=0.5
    )

    encoder_out = torch.randn(batch_size, encoder_dim, H, W)

    # fake captions
    encoded_captions = torch.randint(0, vocab_size, (batch_size, max_len))
    caption_lengths = torch.randint(5, max_len, (batch_size,))

    print("Input shapes:")
    print(f"  Encoder output: {encoder_out.shape}")
    print(f"  Encoded captions: {encoded_captions.shape}")
    print(f"  Caption lengths: {caption_lengths.shape}\n")

    print("Testing forward pass...")
    predictions, alphas = decoder(encoder_out, encoded_captions, caption_lengths)
    print(f"  Predictions shape: {predictions.shape}  (EXPECTED: [B, max_len-1, vocab])")
    print(f"  Alphas shape: {alphas.shape}")

    print("\nTesting generation...")
    single_encoder_out = encoder_out[0:1]
    generated_tokens, gen_alphas = decoder.generate(single_encoder_out, max_len=15)
    print(f"  Generated tokens: {generated_tokens[:10]}... ({len(generated_tokens)} total)")
    if gen_alphas is not None:
        print(f"  Generation alphas shape: {gen_alphas.shape}")

    print("\n" + "="*70)
    print("✅ Decoder test passed!")
    print("="*70)