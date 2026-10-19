import math
from collections import deque
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import LSTM


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.sigmoid()


class ConvolutionModule(nn.Module):
    """
    Convolution Module in SD block.
    """

    def __init__(self, channels: int, depth: int = 2, compress: float = 4, kernel: int = 3):
        super().__init__()
        assert kernel % 2 == 1
        self.depth = abs(depth)
        hidden_size = int(channels / compress)
        norm = lambda d: nn.GroupNorm(1, d)
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            padding = kernel // 2
            mods = [
                norm(channels),
                nn.Conv1d(channels, hidden_size * 2, kernel, padding=padding),
                nn.GLU(1),
                nn.Conv1d(hidden_size, hidden_size, kernel, padding=padding, groups=hidden_size),
                norm(hidden_size),
                Swish(),
                nn.Conv1d(hidden_size, channels, 1),
            ]
            self.layers.append(nn.Sequential(*mods))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class FusionLayer(nn.Module):
    """
    A FusionLayer within the decoder.
    """

    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        if skip is not None:
            x = x + skip
        x = x.repeat(1, 2, 1, 1)
        x = self.conv(x)
        x = F.glu(x, dim=1)
        return x


class SDlayer(nn.Module):
    """
    Sparse Down-sample Layer for splitting into subbands.
    Pre-computes static splits and paddings to eliminate dynamic shape queries and tracer warnings.
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        band_configs: Dict[str, Dict[str, Any]],
        expected_fr: int = 2049,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.strides = []
        self.kernels = []
        for config in band_configs.values():
            self.convs.append(
                nn.Conv2d(channels_in, channels_out, (config["kernel"], 1), (config["stride"], 1), (0, 0))
            )
            self.strides.append(config["stride"])
            self.kernels.append(config["kernel"])

        self.SR_low = band_configs["low"]["SR"]
        self.SR_mid = band_configs["mid"]["SR"]
        self.expected_fr = expected_fr

        # Pre-compute static split points and paddings to eliminate runtime shape indexing & TracerWarnings
        s1 = math.ceil(expected_fr * self.SR_low)
        s2 = math.ceil(expected_fr * (self.SR_low + self.SR_mid))
        self.static_splits = [(0, s1), (s1, s2), (s2, expected_fr)]
        self.static_original_lengths = [s1, s2 - s1, expected_fr - s2]

        self.static_paddings = []
        for (start, end), stride, kernel in zip(self.static_splits, self.strides, self.kernels):
            cur_len = end - start
            if stride == 1:
                tot = kernel - stride
            else:
                tot = (stride - cur_len % stride) % stride
            pl = tot // 2
            pr = tot - pl
            self.static_paddings.append((0, 0, pl, pr))

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[int]]:
        outputs = []
        for conv, (start, end), pad in zip(self.convs, self.static_splits, self.static_paddings):
            extracted = x[:, :, start:end, :]
            padded = F.pad(extracted, pad)
            output = conv(padded)
            outputs.append(output)

        return outputs, self.static_original_lengths


class SUlayer(nn.Module):
    """
    Sparse Up-sample Layer in decoder.
    When configured for vocal extraction, channels_out is 4 (instead of 16).
    Pre-computes static subband slices and symmetric trims.
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        band_configs: Dict[str, Dict[str, Any]],
        lengths: List[int],
        origin_lengths: List[int],
    ):
        super().__init__()
        self.convtrs = nn.ModuleList([
            nn.ConvTranspose2d(channels_in, channels_out, [config["kernel"], 1], [config["stride"], 1])
            for _, config in band_configs.items()
        ])
        self.static_splits = [
            (0, lengths[0]),
            (lengths[0], lengths[0] + lengths[1]),
            (lengths[0] + lengths[1], lengths[0] + lengths[1] + lengths[2]),
        ]
        self.static_trims = []
        for idx, (config, orig_len, (start, end)) in enumerate(
            zip(band_configs.values(), origin_lengths, self.static_splits)
        ):
            in_len = end - start
            stride = config["stride"]
            kernel = config["kernel"]
            out_len = (in_len - 1) * stride + kernel
            dist = abs(orig_len - out_len) // 2
            self.static_trims.append((dist, dist + orig_len))

    def forward(
        self, x: torch.Tensor, lengths: List[int] = None, origin_lengths: List[int] = None
    ) -> torch.Tensor:
        outputs = []
        for convtr, (start, end), (t_start, t_end) in zip(self.convtrs, self.static_splits, self.static_trims):
            out = convtr(x[:, :, start:end, :])
            trimmed_out = out[:, :, t_start:t_end, :]
            outputs.append(trimmed_out)

        return torch.cat(outputs, dim=2)


class SDblock(nn.Module):
    """
    Sparse Down-sample block in encoder.
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        band_configs: Dict[str, Dict[str, Any]],
        conv_config: Dict[str, Any],
        depths: List[int] = [3, 2, 1],
        kernel_size: int = 3,
        expected_fr: int = 2049,
    ):
        super().__init__()
        self.SDlayer = SDlayer(channels_in, channels_out, band_configs, expected_fr=expected_fr)
        self.conv_modules = nn.ModuleList([
            ConvolutionModule(channels_out, depth, **conv_config) for depth in depths
        ])
        self.globalconv = nn.Conv2d(channels_out, channels_out, kernel_size, 1, (kernel_size - 1) // 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
        bands, original_lengths = self.SDlayer(x)
        bands = [
            F.gelu(
                conv(band.permute(0, 2, 1, 3).reshape(-1, band.shape[1], band.shape[3]))
                .view(band.shape[0], band.shape[2], band.shape[1], band.shape[3])
                .permute(0, 2, 1, 3)
            )
            for conv, band in zip(self.conv_modules, bands)
        ]
        lengths = [band.size(-2) for band in bands]
        full_band = torch.cat(bands, dim=2)
        skip = full_band
        output = self.globalconv(full_band)
        return output, skip, lengths, original_lengths


class FeatureConversion(nn.Module):
    def __init__(self, channels: int, inverse: bool):
        super().__init__()
        self.inverse = inverse
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inverse:
            x = x.float()
            x_r = x[:, : self.channels // 2, :, :]
            x_i = x[:, self.channels // 2 :, :, :]
            x = torch.complex(x_r, x_i)
            x = torch.fft.irfft(x, dim=3, norm="ortho")
        else:
            x = x.float()
            x = torch.fft.rfft(x, dim=3, norm="ortho")
            x = torch.cat([x.real, x.imag], dim=1)
        return x


class DualPathRNN(nn.Module):
    def __init__(self, d_model: int, expand: int = 1, bidirectional: bool = True):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = d_model * expand
        self.bidirectional = bidirectional
        self.lstm_layers = nn.ModuleList([
            LSTM(d_model, self.hidden_size, num_layers=1, bidirectional=bidirectional, batch_first=True)
            for _ in range(2)
        ])
        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_size * 2, d_model) for _ in range(2)])
        self.norm_layers = nn.ModuleList([nn.GroupNorm(1, d_model) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        orig = x

        # Frequency path
        x = self.norm_layers[0](x)
        x = x.transpose(1, 3).contiguous().view(B * T, F, C)
        x, _ = self.lstm_layers[0](x)
        x = self.linear_layers[0](x)
        x = x.view(B, T, F, C).transpose(1, 3)
        x = x + orig
        orig = x

        # Time path
        x = self.norm_layers[1](x)
        x = x.transpose(1, 2).contiguous().view(B * F, C, T).transpose(1, 2)
        x, _ = self.lstm_layers[1](x)
        x = self.linear_layers[1](x)
        x = x.transpose(1, 2).contiguous().view(B, F, C, T).transpose(1, 2)
        x = x + orig
        return x


class SeparationNet(nn.Module):
    def __init__(self, channels: int, expand: int = 1, num_layers: int = 6):
        super().__init__()
        self.num_layers = num_layers
        self.dp_modules = nn.ModuleList([
            DualPathRNN(channels * (2 if i % 2 == 1 else 1), expand) for i in range(num_layers)
        ])
        self.feature_conversion = nn.ModuleList([
            FeatureConversion(channels * 2, inverse=(i % 2 != 0)) for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.dp_modules[i](x)
            x = self.feature_conversion[i](x)
        return x


class SCNetVocalExtractor(nn.Module):
    """
    Pruned SCNet model for Apple Neural Engine (ANE) and iOS background execution.
    - Decoupled: Takes pre-computed vDSP real spectrogram [1, 4, 2049, 130].
    - Pruned: Exclusively computes the Vocal Extraction stem.
    - All non-vocal decoder weights (drums, bass, other) are excised.
    """

    def __init__(
        self,
        dims: List[int] = [4, 32, 64, 128],
        band_SR: List[float] = [0.175, 0.392, 0.433],
        band_stride: List[int] = [1, 4, 16],
        band_kernel: List[int] = [3, 4, 16],
        conv_depths: List[int] = [3, 2, 1],
        compress: int = 4,
        conv_kernel: int = 3,
        num_dplayer: int = 6,
        expand: int = 1,
    ):
        super().__init__()
        self.dims = dims
        band_keys = ["low", "mid", "high"]
        self.band_configs = {
            band_keys[i]: {"SR": band_SR[i], "stride": band_stride[i], "kernel": band_kernel[i]}
            for i in range(len(band_keys))
        }
        self.conv_config = {"compress": compress, "kernel": conv_kernel}

        # Pre-compute static subband geometry through all hierarchy levels
        expected_fr = 2049
        current_fr = expected_fr
        enc_specs = []
        for index in range(len(dims) - 1):
            s1 = math.ceil(current_fr * band_SR[0])
            s2 = math.ceil(current_fr * (band_SR[0] + band_SR[1]))
            splits = [(0, s1), (s1, s2), (s2, current_fr)]
            orig_lens = [s1, s2 - s1, current_fr - s2]
            out_lens = []
            for (start, end), stride, kernel in zip(splits, band_stride, band_kernel):
                cur = end - start
                tot = (kernel - stride) if stride == 1 else (stride - cur % stride) % stride
                out_lens.append((cur + tot - kernel) // stride + 1)
            enc_specs.append({
                "expected_fr": current_fr,
                "orig_lens": orig_lens,
                "out_lens": out_lens,
            })
            current_fr = sum(out_lens)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(len(dims) - 1):
            spec = enc_specs[index]
            enc = SDblock(
                channels_in=dims[index],
                channels_out=dims[index + 1],
                band_configs=self.band_configs,
                conv_config=self.conv_config,
                depths=conv_depths,
                expected_fr=spec["expected_fr"],
            )
            self.encoder.append(enc)

        for index in reversed(range(len(dims) - 1)):
            spec = enc_specs[index]
            channels_out = dims[index]  # Always 4 for vocal extraction in final stage
            dec = nn.Sequential(
                FusionLayer(channels=dims[index + 1]),
                SUlayer(
                    channels_in=dims[index + 1],
                    channels_out=channels_out,
                    band_configs=self.band_configs,
                    lengths=spec["out_lens"],
                    origin_lengths=spec["orig_lens"],
                ),
            )
            self.decoder.append(dec)

        self.separation_net = SeparationNet(
            channels=dims[-1],
            expand=expand,
            num_layers=num_dplayer,
        )

        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference forward pass.
        Args:
            x: Input spectrogram [1, 4, 2049, 130] in Float32.
        Returns:
            vocal_spectrogram: Output spectrogram [1, 4, 2049, 130] in Float32.
        """
        save_skip = deque()

        for sd_layer in self.encoder:
            x, skip, _, _ = sd_layer(x)
            save_skip.append(skip)

        x = self.separation_net(x)

        for fusion_layer, su_layer in self.decoder:
            x = fusion_layer(x, save_skip.pop())
            x = su_layer(x)

        # x is already strictly [B, 4, Fr, T] (the isolated Vocal spectrogram)
        return x

    def load_and_prune_weights(self, checkpoint_path: str):
        """
        Loads weights from a 4-stem SCNet checkpoint, slicing out only the vocal extraction channels.
        """
        print(f"Loading weights from checkpoint: {checkpoint_path}")
        raw = torch.load(checkpoint_path, map_location="cpu")
        sd = raw.get("state_dict", raw.get("model", raw))

        # Clean module. or model. prefixes
        cleaned_sd = {}
        for k, v in sd.items():
            k_clean = k
            if k_clean.startswith("module."):
                k_clean = k_clean[7:]
            if k_clean.startswith("model."):
                k_clean = k_clean[6:]
            cleaned_sd[k_clean] = v

        model_sd = self.state_dict()
        pruned_sd = {}

        # Slicing vocal channels (indices 12..16) from the final decoder layer
        vocal_slice = slice(12, 16)
        final_layer_prefix = f"decoder.{len(self.dims) - 2}.1.convtrs."

        for key, tensor in cleaned_sd.items():
            if key in model_sd:
                # Check for final SUlayer convtrs projection
                if any(key.startswith(f"{final_layer_prefix}{i}.") for i in range(3)):
                    if "weight" in key:
                        # Shape: [in_channels, 16, kernel, 1] -> slice to [in_channels, 4, kernel, 1]
                        pruned_sd[key] = tensor[:, vocal_slice, :, :]
                        print(f"  [PRUNED] {key}: {tensor.shape} -> {pruned_sd[key].shape} (Vocals only)")
                    elif "bias" in key:
                        # Shape: [16] -> slice to [4]
                        pruned_sd[key] = tensor[vocal_slice]
                        print(f"  [PRUNED] {key}: {tensor.shape} -> {pruned_sd[key].shape} (Vocals only)")
                    else:
                        pruned_sd[key] = tensor
                else:
                    pruned_sd[key] = tensor

        missing, unexpected = self.load_state_dict(pruned_sd, strict=True)
        print(f"Successfully loaded vocal-pruned weights (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

