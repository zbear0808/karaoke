import os
import sys
import warnings
import torch
import numpy as np
from typing import Optional

# Suppress TorchScript deprecation notices during CoreML tracing
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")

try:
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )
except ImportError:
    ct = None

from config import MobileQuantizationConfig
from scnet_pruner import MobileSCNetWrapper


class CoreMLQuantizationEngine:
    """
    Translates traced PyTorch SCNet modules into INT8-quantized, ANE-compatible CoreML packages.
    """

    def __init__(self, config: MobileQuantizationConfig = MobileQuantizationConfig()):
        self.config = config

    def export_and_quantize(
        self,
        wrapped_pytorch_model: MobileSCNetWrapper,
        output_package_path: str = "SCNetInstrumental.mlpackage",
        compile_to_mlmodelc: bool = True
    ) -> Optional[str]:
        """
        Runs the full CoreML export and INT8 post-training quantization workflow.
        """
        shape = self.config.input_shape
        print(f"=== Step 1: Tracing PyTorch Graph (Static Shape: {shape}) ===")

        dummy_input = torch.randn(*shape, dtype=torch.float32)
        wrapped_pytorch_model.eval()

        with torch.no_grad():
            traced_model = torch.jit.trace(wrapped_pytorch_model, dummy_input)
            traced_model.eval()

        fallback_torchscript_path = output_package_path.replace(".mlpackage", "_traced.pt")
        print(f"Saving traced TorchScript model to: {fallback_torchscript_path}")
        traced_model.save(fallback_torchscript_path)

        if ct is None:
            print("[NOTE] coremltools not available in current Python environment.")
            return fallback_torchscript_path

        input_type = ct.TensorType(
            name="input_spectrogram",
            shape=ct.Shape(shape=shape),
            dtype=np.float32
        )
        output_type = ct.TensorType(name="output_spectrogram")

        target_deployment = (
            ct.target.iOS17 if self.config.minimum_ios_deployment_target == "iOS17" else ct.target.iOS18
        )

        try:
            print("=== Step 2: Converting to CoreML MIL mlprogram (Float16) ===")
            mlmodel = ct.convert(
                traced_model,
                inputs=[input_type],
                outputs=[output_type],
                compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=target_deployment,
                convert_to="mlprogram"
            )

            print(f"=== Step 3: Applying INT8 Post-Training Weight Quantization ({self.config.quantization_mode}) ===")
            opt_config = OptimizationConfig(
                global_config=OpLinearQuantizerConfig(
                    mode=self.config.quantization_mode,
                    dtype=self.config.weight_dtype
                )
            )
            quantized_mlmodel = linear_quantize_weights(mlmodel, config=opt_config)

            print(f"=== Step 4: Saving Quantized Package to {output_package_path} ===")
            quantized_mlmodel.save(output_package_path)
            print("MLPackage export completed successfully.")

            if compile_to_mlmodelc:
                print("=== Step 5: Compiling MLPackage to .mlmodelc ===")
                compile_dir = os.path.dirname(output_package_path) or "."
                cmd = f"xcrun coremlcompiler compile {output_package_path} {compile_dir}"
                print(f"Executing: {cmd}")
                ret = os.system(cmd)
                if ret == 0:
                    print("Successfully generated pre-compiled .mlmodelc bundle.")
                else:
                    print("[NOTE] xcrun coremlcompiler not found (standard when running on non-macOS host). Run on Mac before packaging.")

            return output_package_path

        except RuntimeError as e:
            if "BlobWriter not loaded" in str(e):
                print("\n[NOTE] CoreML 'mlprogram' packaging requires Apple's native C++ BlobWriter library (distributed only for macOS/Linux).")
                print("Exporting as CoreML NeuralNetwork format (.mlmodel) which is fully supported on Windows...")
                mlmodel_path = output_package_path.replace(".mlpackage", ".mlmodel")
                mlmodel_nn = ct.convert(
                    traced_model,
                    inputs=[input_type],
                    convert_to="neuralnetwork"
                )
                mlmodel_nn.save(mlmodel_path)
                print(f"[SUCCESS] Exported Windows-compatible CoreML model: {mlmodel_path}")
                print(f"[SUCCESS] Exported cross-platform TorchScript model: {fallback_torchscript_path}")
                print("[INFO] For final iOS ANE deployment (.mlpackage / .mlmodelc with INT8 weights), compile on macOS or macOS CI runner.")
                return mlmodel_path
            else:
                raise e

