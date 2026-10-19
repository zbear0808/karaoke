import os
import sys
import math
import argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from scnet_vocal_model import SCNetVocalExtractor


def separate_audio_track(
    audio_path: str,
    checkpoint_path: str = None,
    output_dir: str = "test_audio/output",
    chunk_frames: int = 130,  # 130 frames = ~3.0s static window matching CoreML/ANE graph
    hop_length: int = 1024,
    n_fft: int = 4096,
    overlap_ratio: float = 0.5,
    max_duration: float = None,
    device: str = "cpu"
):
    """
    Separates vocal and instrumental tracks from a real audio file using
    the pruned SCNetVocalExtractor model.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    # 1. Resolve Checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "checkpoints", "scnet_base.ckpt")
        )

    print(f"==================================================================")
    print(f"      SCNet Vocal Separation Audio Test                          ")
    print(f"==================================================================")
    print(f"Input Audio Track : {audio_path}")
    print(f"Model Checkpoint  : {checkpoint_path}")

    # 2. Load Model
    model = SCNetVocalExtractor()
    model.load_and_prune_weights(checkpoint_path)
    model.to(device)
    model.eval()

    # 3. Load Audio (stereo, 44.1 kHz)
    audio_data, sr = sf.read(audio_path)
    if audio_data.ndim == 1:
        # Mono to stereo
        audio_data = np.stack([audio_data, audio_data], axis=-1)
    elif audio_data.shape[1] > 2:
        audio_data = audio_data[:, :2]

    num_samples = audio_data.shape[0]
    duration_sec = num_samples / sr
    print(f"Audio Properties  : {num_samples} samples, {duration_sec:.2f}s, {sr} Hz, {audio_data.shape[1]} channels")

    # Shape: [1, 2, num_samples]
    waveform = torch.from_numpy(audio_data.T).float().unsqueeze(0).to(device)

    # Resample to 44.1 kHz if necessary
    target_sr = 44100
    if sr != target_sr:
        print(f"Resampling audio from {sr} Hz to {target_sr} Hz...")
        import torchaudio.transforms as T
        resampler = T.Resample(sr, target_sr).to(device)
        waveform = resampler(waveform)
        sr = target_sr
        num_samples = waveform.shape[-1]
        duration_sec = num_samples / sr

    if max_duration is not None and duration_sec > max_duration:
        max_samples = int(max_duration * sr)
        waveform = waveform[:, :, :max_samples]
        num_samples = max_samples
        duration_sec = max_duration
        print(f"Testing on first {max_duration:.1f}s preview window...")

    # 4. SCNet Exact STFT Transformation (No Hann window distortion)
    # SCNet ensures that T is even after STFT so RFFT can be applied in DualPathRNN
    padding = hop_length - waveform.shape[-1] % hop_length
    if (waveform.shape[-1] + padding) // hop_length % 2 == 0:
        padding += hop_length
    waveform_padded = F.pad(waveform, (0, padding))

    stft_cfg = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": n_fft,
        "center": True,
        "normalized": True
    }

    L = waveform_padded.shape[-1]
    x_in = waveform_padded.reshape(-1, L)
    spec_complex = torch.stft(x_in, **stft_cfg, return_complex=True)
    spec_real = torch.view_as_real(spec_complex)
    # Interleave to [1, 4, 2049, T]: [L_real, L_imag, R_real, R_imag]
    in_spec = spec_real.permute(0, 3, 1, 2).reshape(1, 4, spec_complex.shape[1], spec_complex.shape[2])

    print(f"Executing SCNet forward pass on spectrogram: {tuple(in_spec.shape)}...")
    with torch.no_grad():
        out_spec = model(in_spec)

    # Reconstruct stereo complex vocals
    v_L = torch.complex(out_spec[:, 0], out_spec[:, 1])
    v_R = torch.complex(out_spec[:, 2], out_spec[:, 3])
    v_complex = torch.cat([v_L, v_R], dim=0)  # [2, 2049, T]

    # Inverse STFT directly matching training config
    vocal_waveform_padded = torch.istft(v_complex, **stft_cfg, length=waveform_padded.shape[-1])
    vocal_waveform = vocal_waveform_padded.unsqueeze(0)[:, :, :-padding]
    mixture_waveform = waveform[:, :, :vocal_waveform.shape[-1]]
    instrumental_waveform = mixture_waveform - vocal_waveform

    # Save to disk
    vocal_np = vocal_waveform.squeeze(0).cpu().numpy().T
    inst_np = instrumental_waveform.squeeze(0).cpu().numpy().T
    orig_np = mixture_waveform.squeeze(0).cpu().numpy().T

    vocal_path = os.path.join(output_dir, f"{base_name}_vocals.wav")
    inst_path = os.path.join(output_dir, f"{base_name}_instrumental.wav")
    orig_path = os.path.join(output_dir, f"{base_name}_original.wav")

    sf.write(vocal_path, vocal_np, sr)
    sf.write(inst_path, inst_np, sr)
    sf.write(orig_path, orig_np, sr)

    # Compute RMS energy metrics
    rms_orig = np.sqrt(np.mean(orig_np ** 2))
    rms_vocal = np.sqrt(np.mean(vocal_np ** 2))
    rms_inst = np.sqrt(np.mean(inst_np ** 2))

    print(f"\n--- Separation Complete ---")
    print(f"Original Audio RMS Energy   : {rms_orig:.4f}")
    print(f"Extracted Vocals RMS Energy : {rms_vocal:.4f} (Saved to: {vocal_path})")
    print(f"Extracted Karaoke RMS Energy: {rms_inst:.4f} (Saved to: {inst_path})")
    print(f"Energy Ratio (Vocals/Mix)   : {100.0 * rms_vocal / (rms_orig + 1e-8):.1f}%")
    print(f"==================================================================\n")

    return {
        "vocals_path": vocal_path,
        "instrumental_path": inst_path,
        "original_path": orig_path,
        "rms_vocals": float(rms_vocal),
        "rms_instrumental": float(rms_inst),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test vocal extraction on real audio file.")
    parser.add_argument("--audio", type=str, default="test_audio/audio_example.mp3", help="Input audio path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (default: scnet_base.ckpt)")
    parser.add_argument("--out-dir", type=str, default="test_audio/output", help="Output directory")
    parser.add_argument("--max-duration", type=float, default=None, help="Max duration in seconds to process (optional)")

    args = parser.parse_args()
    separate_audio_track(
        audio_path=args.audio,
        checkpoint_path=args.checkpoint,
        output_dir=args.out_dir,
        max_duration=args.max_duration
    )

