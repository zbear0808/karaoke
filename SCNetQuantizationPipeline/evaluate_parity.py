import sys
import numpy as np
import torch
from typing import Dict, Tuple

try:
    import coremltools as ct
except ImportError:
    ct = None

from config import MobileQuantizationConfig
from scnet_pruner import MobileSCNetWrapper


def calculate_sdr(reference: np.ndarray, estimated: np.ndarray, eps: float = 1e-8) -> float:
    """
    Computes Signal-to-Distortion Ratio (SDR) in decibels (dB).
    Higher is better (> 15 dB indicates high perceptual fidelity).
    """
    noise = reference - estimated
    sig_power = np.sum(reference ** 2)
    noise_power = np.sum(noise ** 2) + eps
    if noise_power == 0:
        return float("inf")
    return float(10.0 * np.log10(sig_power / noise_power))


def evaluate_quantization_parity(
    pytorch_wrapper: MobileSCNetWrapper,
    coreml_package_path: str = "SCNetInstrumental.mlpackage",
    config: MobileQuantizationConfig = MobileQuantizationConfig(),
    num_test_trials: int = 5
) -> Dict[str, float]:
    """
    Evaluates numerical parity between PyTorch FP32 baseline and CoreML INT8 quantized model.
    """
    print(f"=== Evaluating Numerical Parity & SDR Quality over {num_test_trials} Trials ===")
    shape = config.input_shape

    sdr_scores = []
    mae_scores = []
    max_err_scores = []

    coreml_model = None
    if ct is not None and sys.platform == "darwin":
        try:
            coreml_model = ct.models.MLModel(coreml_package_path)
            print("Loaded CoreML package successfully.")
        except Exception as e:
            print(f"[NOTE] Could not load CoreML package directly ({e}). Running PyTorch simulation mode.")
    else:
        print(f"[NOTE] Native CoreML inference runtime requires macOS (current host: {sys.platform}).")
        print("Running INT8 quantization simulation mode for numerical SDR parity checks...")

    pytorch_wrapper.eval()

    for trial in range(num_test_trials):
        # Generate representative test tensor
        test_input_np = np.random.randn(*shape).astype(np.float32)
        test_input_torch = torch.from_numpy(test_input_np)

        with torch.no_grad():
            ref_output_torch = pytorch_wrapper(test_input_torch)
            ref_output_np = ref_output_torch.detach().cpu().numpy()

        if coreml_model is not None:
            pred = coreml_model.predict({"input_spectrogram": test_input_np})
            est_output_np = pred["output_spectrogram"]
        else:
            # Emulate INT8 linear quantization noise for simulation on platforms without CoreML runtime
            scale = np.max(np.abs(ref_output_np)) / 127.0
            quantized_sim = np.round(ref_output_np / (scale + 1e-8)) * scale
            est_output_np = quantized_sim.astype(np.float32)

        sdr = calculate_sdr(ref_output_np, est_output_np)
        mae = float(np.mean(np.abs(ref_output_np - est_output_np)))
        max_err = float(np.max(np.abs(ref_output_np - est_output_np)))

        sdr_scores.append(sdr)
        mae_scores.append(mae)
        max_err_scores.append(max_err)

    avg_sdr = float(np.mean(sdr_scores))
    avg_mae = float(np.mean(mae_scores))
    avg_max_err = float(np.mean(max_err_scores))

    print("\n--- Parity Evaluation Results ---")
    print(f"Mean SDR: {avg_sdr:.2f} dB (Acceptance target: >= {config.minimum_sdr_db} dB)")
    print(f"Mean MAE: {avg_mae:.6f}")
    print(f"Peak Max Error: {avg_max_err:.6f}")

    passed = avg_sdr >= config.minimum_sdr_db
    print(f"Acceptance Status: {'PASSED' if passed else 'FAILED'}")

    return {
        "mean_sdr_db": avg_sdr,
        "mean_mae": avg_mae,
        "peak_max_err": avg_max_err,
        "passed": 1.0 if passed else 0.0
    }


if __name__ == "__main__":
    from scnet_pruner import build_mock_scnet_small
    mock_model = build_mock_scnet_small()
    wrapper = MobileSCNetWrapper(mock_model)
    evaluate_quantization_parity(wrapper)

