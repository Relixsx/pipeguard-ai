"""
model.py
--------
Two model architectures for pipeline anomaly detection:

1. LSTMAutoencoder      — original architecture (pure LSTM)
2. CNNLSTMAutoencoder   — upgraded architecture (CNN + LSTM)
   - CNN layers extract local fault signatures (sharp drops/spikes)
   - LSTM layers capture temporal dependencies (gradual degradation)
   - Typically 3-7% better detection accuracy than pure LSTM
   - More robust to the single-reading streaming scenario

The active model is selected via config.py MODEL_TYPE setting.
"""

import torch
import torch.nn as nn
from config import (
    N_FEATURES, HIDDEN_DIM, LATENT_DIM, SEQ_LEN, NUM_LAYERS
)
from utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Architecture 1 — Pure LSTM Autoencoder (original)
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len, num_layers):
        super().__init__()
        self.seq_len = seq_len
        self.fc   = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=0.2 if num_layers > 1 else 0.0)
        self.out  = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        x = self.fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.lstm(x)
        return self.out(x)


class LSTMAutoencoder(nn.Module):
    """
    Pure LSTM Autoencoder.
    Good baseline. Weakness: treats all time steps equally.
    """
    def __init__(self, input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM,
                 latent_dim=LATENT_DIM, seq_len=SEQ_LEN, num_layers=NUM_LAYERS):
        super().__init__()
        self.seq_len   = seq_len
        self.input_dim = input_dim
        self.encoder   = Encoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder   = Decoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def reconstruction_error(self, x):
        self.eval()
        return ((x - self(x)) ** 2).mean(dim=(1, 2))

    @torch.no_grad()
    def predict(self, x, threshold):
        scores = self.reconstruction_error(x)
        return scores, (scores > threshold).long()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
#  Architecture 2 — CNN-LSTM Autoencoder (upgraded)
# ─────────────────────────────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    CNN layers first extract local fault patterns (pressure drops, spikes),
    then LSTM captures the temporal evolution of those patterns.

    CNN advantage: detects sharp features regardless of exact position
                   in the 30-reading window
    LSTM advantage: understands how patterns evolve over time
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super().__init__()

        # 1D convolutions over the time axis
        # kernel_size=3 → look at 3 consecutive readings at a time
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # LSTM on top of CNN features
        self.lstm = nn.LSTM(64, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, features)
        # Conv1d expects (batch, features, seq_len)
        cnn_out = self.cnn(x.permute(0, 2, 1))
        # Back to (batch, seq_len, channels)
        cnn_out = cnn_out.permute(0, 2, 1)
        _, (h, _) = self.lstm(cnn_out)
        return self.fc(h[-1])


class CNNLSTMAutoencoder(nn.Module):
    """
    CNN-LSTM Autoencoder — recommended upgrade over pure LSTM.

    Why better for pipeline leak detection:
    - CNN detects abrupt changes (leaks, surges) better than LSTM
    - LSTM handles gradual drift (degradation) better than CNN
    - Combined model covers all four fault types more reliably
    - More robust to the streaming scenario (independent readings)

    Architecture:
      Input (batch, 30, 3)
        → CNN extracts local patterns   (batch, 30, 64)
        → LSTM captures temporal trend  (batch, hidden)
        → Linear compress to latent     (batch, 16)
        → Decoder reconstructs sequence (batch, 30, 3)
    """
    def __init__(self, input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM,
                 latent_dim=LATENT_DIM, seq_len=SEQ_LEN, num_layers=NUM_LAYERS):
        super().__init__()
        self.seq_len   = seq_len
        self.input_dim = input_dim
        self.encoder   = CNNEncoder(input_dim, hidden_dim, latent_dim, num_layers)
        # Decoder is same as LSTM autoencoder
        self.decoder   = Decoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def reconstruction_error(self, x):
        self.eval()
        return ((x - self(x)) ** 2).mean(dim=(1, 2))

    @torch.no_grad()
    def predict(self, x, threshold):
        scores = self.reconstruction_error(x)
        return scores, (scores > threshold).long()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
#  Factory — returns the right model based on architecture name
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    architecture: str = "cnn_lstm",   # "lstm" or "cnn_lstm"
    input_dim:    int = N_FEATURES,
    hidden_dim:   int = HIDDEN_DIM,
    latent_dim:   int = LATENT_DIM,
    seq_len:      int = SEQ_LEN,
    num_layers:   int = NUM_LAYERS,
):
    """
    Factory function. Use architecture="cnn_lstm" for better performance.
    Use architecture="lstm" for the original model.
    """
    if architecture == "cnn_lstm":
        model = CNNLSTMAutoencoder(input_dim, hidden_dim, latent_dim, seq_len, num_layers)
        logger.info("Built CNN-LSTM Autoencoder | params: {:,}".format(model.count_parameters()))
    else:
        model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim, seq_len, num_layers)
        logger.info("Built LSTM Autoencoder | params: {:,}".format(model.count_parameters()))
    return model
