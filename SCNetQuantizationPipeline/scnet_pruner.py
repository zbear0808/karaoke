import torch
import torch.nn as nn
from typing import Dict, Any, Union
from config import MobileQuantizationConfig


class MobileSCNetWrapper(nn.Module):
    """
    Wraps SCNet PyTorch model for Apple Neural Engine (ANE) compatibility:
    - Eliminates torch.stft and torch.istft.
    - Locks static tensor shape [1, 4, 2049, 130].
    - Prunes multi-stem decoder down to Instrumental (or Vocal).
    """

    def __init__(
        self,
        base_model: nn.Module,
        target_stem: str = "instrumental",
        config: MobileQuantizationConfig = MobileQuantizationConfig()
    ):
        super().__init__()
        self.base_model = base_model
        self.target_stem = target_stem.lower()
        self.expected_shape = config.input_shape

        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with static shape assertions.

        Args:
            input_spectrogram: Tensor of shape [1, 4, 2049, 130] in Float32.
                               Channels represent [L_real, L_imag, R_real, R_imag].

        Returns:
            output_spectrogram: Tensor of shape [1, 4, 2049, 130] in Float32.
        """
        assert input_spectrogram.shape == self.expected_shape, (
            f"Expected static input shape {self.expected_shape}, got {input_spectrogram.shape}"
        )

        # Execute base model core (handles various community SCNet forward interfaces)
        if hasattr(self.base_model, "separate_spectrogram"):
            stems = self.base_model.separate_spectrogram(input_spectrogram)
        elif hasattr(self.base_model, "forward_core"):
            stems = self.base_model.forward_core(input_spectrogram)
        else:
            stems = self.base_model(input_spectrogram)

        # Extract target stem
        if isinstance(stems, dict):
            if self.target_stem in stems:
                out = stems[self.target_stem]
            elif "accompaniment" in stems and self.target_stem == "instrumental":
                out = stems["accompaniment"]
            elif "vocals" in stems and self.target_stem == "vocals":
                out = stems["vocals"]
            else:
                out = list(stems.values())[0]
        elif isinstance(stems, (list, tuple)):
            out = stems[0]
        elif isinstance(stems, torch.Tensor):
            if stems.dim() == 5:
                # Shape: [Batch, NumStems, 4, Freq, Time]
                stem_idx = 0 if self.target_stem == "instrumental" else 1
                out = stems[:, stem_idx, :, :, :]
            else:
                out = stems
        else:
            raise ValueError(f"Unrecognized stems output type: {type(stems)}")

        return out


def build_mock_scnet_small() -> nn.Module:
    """
    Builds a lightweight structural mock model matching the SCNet subband conv topology.
    Useful for offline testing and verifying quantization passes without external checkpoints.
    """
    class MockSCNetSmall(nn.Module):
        def __init__(self):
            super().__init__()
            # Lightweight subband conv-residual blocks
            self.encoder = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=(3, 3), stride=1, padding=1),
                nn.GroupNorm(4, 32),
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
                nn.GroupNorm(4, 32),
                nn.GELU(),
            )
            self.bottleneck = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(1, 1)),
                nn.GELU(),
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
                nn.GroupNorm(4, 32),
                nn.GELU(),
                nn.Conv2d(32, 4, kernel_size=(3, 3), stride=1, padding=1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            latent = self.encoder(x)
            bottleneck = self.bottleneck(latent)
            mask = self.decoder(bottleneck)
            return x * mask

    return MockSCNetSmall()

