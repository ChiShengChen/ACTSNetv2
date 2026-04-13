import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqLensAttention(nn.Module):
    """
    Frequency-domain attention mechanism.

    Computes attention weights in the frequency domain via FFT,
    allowing the model to explicitly attend to specific frequency components
    that are most discriminative for MDD classification.

    Replaces the original simple AC = Softmax(X) * X.

    Architecture:
        1. FFT -> frequency representation
        2. Learnable frequency filter (complex-valued attention)
        3. Frequency-domain gating
        4. iFFT -> back to time domain
        5. Residual connection with original signal

    Args:
        d_model: model dimension
        dropout: dropout rate
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Learnable frequency filter — complex-valued
        self.freq_gate_real = nn.Linear(d_model, d_model)
        self.freq_gate_imag = nn.Linear(d_model, d_model)

        # Frequency importance scorer
        self.freq_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid(),
        )

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, N, d_model) — batch, sequence_len (patches), features
        returns: (B, N, d_model) — frequency-attended representation
        """
        residual = x

        # 1. FFT along the sequence dimension
        x_freq = torch.fft.rfft(x, dim=1)  # (B, N//2+1, d_model) complex

        # 2. Compute frequency attention weights
        freq_magnitude = torch.abs(x_freq)  # (B, F, d_model)
        freq_attention = self.freq_scorer(freq_magnitude)  # (B, F, d_model) in [0,1]

        # 3. Apply learnable complex filter
        real_part = self.freq_gate_real(x_freq.real)
        imag_part = self.freq_gate_imag(x_freq.imag)
        x_filtered = torch.complex(real_part, imag_part)

        # 4. Gate by frequency attention
        x_filtered = x_filtered * freq_attention

        # 5. iFFT back to time domain
        x_out = torch.fft.irfft(x_filtered, n=x.shape[1], dim=1)  # (B, N, d_model)

        # 6. Project + residual
        x_out = self.dropout(self.out_proj(x_out))
        x_out = self.layer_norm(x_out + residual)

        return x_out

    def get_freq_attribution(self, x):
        """
        Get frequency-domain attribution map for interpretability.
        Returns per-frequency importance scores.

        x: (B, N, d_model)
        returns: (B, n_freq_bins, d_model) — attribution scores
        """
        with torch.no_grad():
            x_freq = torch.fft.rfft(x, dim=1)
            freq_magnitude = torch.abs(x_freq)
            freq_attention = self.freq_scorer(freq_magnitude)
        return freq_attention
