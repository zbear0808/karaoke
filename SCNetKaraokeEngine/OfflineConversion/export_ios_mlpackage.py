import os
import sys
import torch
import numpy as np

try:
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )
except ImportError:
    ct = None

from scnet_core_wrapper import SCNetCoreANEWrapper, build_mock_scnet_small


def export_scnet_to_ios(
    model_or_checkpoint=None,
    output_mlpackage_path: str = "SCNetInstrumental.mlpackage",
    target_ios_version: str = "iOS17"
):
    """
    Exports SCNet PyTorch model to an iOS-optimized CoreML .mlpackage:
    1. Traces the model using torch.jit.trace with fixed [1, 4, 2049, 130] dummy tensor.
    2. Runs coremltools.convert targeting iOS 17/18 with Float16 compute precision.
    3. Applies post-training weight quantization (linear_quantize_weights, INT8) to shrink weights to ~5-8 MB.
    """
    print("=== Step 1: Preparing PyTorch Model Wrapper ===")
    if model_or_checkpoint is None:
        print("No checkpoint provided. Building reference SCNet-Small topology mock...")
        base_model = build_mock_scnet_small()
    elif isinstance(model_or_checkpoint, str):
        print(f"Loading checkpoint from: {model_or_checkpoint}")
        checkpoint = torch.load(model_or_checkpoint, map_location="cpu")
        # Instantiate model based on checkpoint state dict if available
        base_model = build_mock_scnet_small()
        if "state_dict" in checkpoint:
            base_model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            base_model.load_state_dict(checkpoint, strict=False)
    else:
        base_model = model_or_checkpoint

    wrapper = SCNetCoreANEWrapper(base_model, target_stem="instrumental")
    wrapper.eval()

    # Fixed dummy input [1, 4, 2049, 130]
    dummy_input = torch.randn(1, 4, 2049, 130, dtype=torch.float32)

    print("=== Step 2: Tracing PyTorch Graph ===")
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, dummy_input)
        traced_model.eval()

    if ct is None:
        print("[WARNING] coremltools is not installed in the current environment.")
        print("Saving traced TorchScript model to SCNetInstrumental_traced.pt for compilation on macOS.")
        traced_model.save("SCNetInstrumental_traced.pt")
        return

    print("=== Step 3: Converting to CoreML MIL ===")
    input_shape = ct.Shape(shape=(1, 4, 2049, 130))
    inputs = [
        ct.TensorType(
            name="input_spectrogram",
            shape=input_shape,
            dtype=np.float32
        )
    ]

    outputs = [
        ct.TensorType(
            name="output_spectrogram"
        )
    ]

    target_deployment = ct.target.iOS17 if target_ios_version == "iOS17" else ct.target.iOS18

    mlmodel = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=target_deployment,
        convert_to="mlprogram"
    )

    print("=== Step 4: Applying INT8 Weight Quantization ===")
    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8"
        )
    )
    quantized_mlmodel = linear_quantize_weights(mlmodel, config=config)

    print(f"Saving quantized MLPackage to: {output_mlpackage_path}")
    quantized_mlmodel.save(output_mlpackage_path)
    print("CoreML package successfully exported.")

    print("=== Step 5: Pre-compiling to .mlmodelc ===")
    compiled_output_dir = os.path.dirname(output_mlpackage_path) or "."
    compile_cmd = f"xcrun coremlcompiler compile {output_mlpackage_path} {compiled_output_dir}"
    print(f"Running: {compile_cmd}")
    compile_status = os.system(compile_cmd)
    if compile_status == 0:
        print("Successfully generated pre-compiled .mlmodelc bundle.")
    else:
        print("[NOTE] xcrun coremlcompiler not available on this host. Run the command on macOS before packaging the app.")



if __name__ == "__main__":
    checkpoint_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else "SCNetInstrumental.mlpackage"
    export_scnet_to_ios(checkpoint_arg, out_arg)

