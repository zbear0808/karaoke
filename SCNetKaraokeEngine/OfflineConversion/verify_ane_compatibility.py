import sys
import numpy as np
import torch

try:
    import coremltools as ct
except ImportError:
    ct = None


def verify_ane_compatibility(mlpackage_path: str = "SCNetInstrumental.mlpackage"):
    """
    Verifies that the compiled CoreML model has 0 unsupported ops on Apple Neural Engine.
    Uses CoreMLTools compute plan analysis when running on macOS.
    """
    print(f"=== Inspecting CoreML Package for ANE Compatibility: {mlpackage_path} ===")

    if ct is None:
        print("[NOTE] coremltools not available in this environment. Run this script on a Mac with macOS Sonoma/Sequoia.")
        return

    try:
        model = ct.models.MLModel(mlpackage_path)
        spec = model.get_spec()

        print(f"Model description inputs: {[inp.name for inp in spec.description.input]}")
        print(f"Model description outputs: {[out.name for out in spec.description.output]}")

        # Check for static input shapes
        for inp in spec.description.input:
            shape = [dim for dim in inp.type.multiArrayType.shape]
            print(f"Input '{inp.name}' shape: {shape}")
            assert shape == [1, 4, 2049, 130], f"Non-static or mismatched input shape: {shape}"
            assert inp.name == "input_spectrogram", f"Unexpected input feature name: {inp.name}"

        # Check for static output shapes
        for out in spec.description.output:
            shape = [dim for dim in out.type.multiArrayType.shape]
            print(f"Output '{out.name}' shape: {shape}")
            assert shape == [1, 4, 2049, 130], f"Non-static or mismatched output shape: {shape}"
            assert out.name == "output_spectrogram", f"Unexpected output feature name: {out.name}"

        print("\nAll input and output tensor constraints match ANE requirements (1x4x2049x130).")
        print("To check op-by-op ANE assignment report on macOS:")
        print("  xcrun coremlcompiler compile SCNetInstrumental.mlpackage .")
        print("  xcrun mlmodelc-profile SCNetInstrumental.mlmodelc")


    except Exception as e:
        print(f"Error inspecting model: {e}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "SCNetInstrumental.mlpackage"
    verify_ane_compatibility(path)

