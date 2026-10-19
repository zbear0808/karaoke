from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class MobileQuantizationConfig:
    """
    Configuration parameters for mobile ANE SCNet export and quantization.
    Matches constants defined in iOSApp/Config/SCNetEngineConfig.swift.
    """
    sample_rate: int = 44100
    channels: int = 2
    n_fft: int = 4096
    hop_length: int = 1024
    freq_bins: int = 2049
    stft_frames: int = 130
    window_duration_seconds: float = 3.0
    step_duration_seconds: float = 1.5

    # Exact sample counts
    window_samples: int = 132300   # 3.0 * 44100
    step_samples: int = 66150      # 1.5 * 44100
    padded_window_samples: int = 133120  # (130 - 1) * 1024 + 4096

    # Tensor shape: [Batch, Channels (L_re, L_im, R_re, R_im), FreqBins, TimeFrames]
    input_shape: Tuple[int, int, int, int] = (1, 4, 2049, 130)

    # Quantization settings
    weight_dtype: str = "int8"
    quantization_mode: str = "linear_symmetric"
    compute_precision: str = "float16"
    minimum_ios_deployment_target: str = "iOS17"

    # Quality acceptance thresholds
    minimum_sdr_db: float = 10.0
    max_sdr_loss_db: float = 0.5

