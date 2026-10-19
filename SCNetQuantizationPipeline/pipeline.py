import argparse
import sys
import os
import warnings
import torch

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")

from config import MobileQuantizationConfig
from scnet_vocal_model import SCNetVocalExtractor
from quantize_engine import CoreMLQuantizationEngine
from evaluate_parity import evaluate_quantization_parity


def run_vocal_pipeline(
    model_variant: str = "base",
    checkpoint_path: str = None,
    output_package: str = "SCNetVocals.mlpackage",
    evaluate: bool = True
):
    """
    End-to-end Vocal Extraction Pipeline:
    Ingests SCNet Base or Small checkpoint, excises all non-vocal weights,
    and exports a static [1, 4, 2049, 130] CoreML / TorchScript vocal extractor.
    """
    config = MobileQuantizationConfig()

    print("==================================================================")
    print(f"      SCNet Vocal Extraction & Quantization Pipeline ({model_variant.upper()})      ")
    print("==================================================================")

    # 1. Resolve Checkpoint Path
    if checkpoint_path is None:
        default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
        filename = f"scnet_{model_variant}.ckpt"
        checkpoint_path = os.path.join(default_dir, filename)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Run download_checkpoints.py first.")

    # 2. Instantiate and Prune to Vocal-Only Architecture
    print(f"[INFO] Initializing SCNetVocalExtractor and pruning non-vocal weights...")
    vocal_model = SCNetVocalExtractor()
    vocal_model.load_and_prune_weights(checkpoint_path)

    # 3. Export and Quantize
    engine = CoreMLQuantizationEngine(config=config)
    exported_path = engine.export_and_quantize(
        wrapped_pytorch_model=vocal_model,
        output_package_path=output_package,
        compile_to_mlmodelc=True
    )

    # 4. Numerical Parity Evaluation
    if evaluate:
        results = evaluate_quantization_parity(
            pytorch_wrapper=vocal_model,
            coreml_package_path=exported_path,
            config=config
        )
        if results["passed"] < 1.0:
            print("[WARNING] Quantization SDR did not meet configured quality threshold.")
        else:
            print("[SUCCESS] Vocal extractor passed all mobile parity checks.")

    print("\nVocal extraction pipeline execution complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and quantize SCNet vocal extraction model for iOS ANE.")
    parser.add_argument("--model", type=str, default="base", choices=["base", "small"], help="Model variant: base or small")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path (optional)")
    parser.add_argument("--out", type=str, default=None, help="Output .mlpackage filename (optional)")
    parser.add_argument("--no-eval", action="store_true", help="Skip numerical parity evaluation")

    args = parser.parse_args()
    out_pkg = args.out or f"SCNetVocals_{args.model}.mlpackage"
    run_vocal_pipeline(
        model_variant=args.model,
        checkpoint_path=args.checkpoint,
        output_package=out_pkg,
        evaluate=not args.no_eval
    )
