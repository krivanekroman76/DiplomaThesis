# Test script for separators
print("Testing imports...")

try:
    from spleeter_separator import SpleeterSeparator
    spleeter_sep = SpleeterSeparator()
    print("✓ Spleeter ready")
except Exception as e:
    print(f"✗ Spleeter error: {e}")

try:
    from demucs_separator import DemucsSeparator
    demucs_sep = DemucsSeparator()
    print("✓ Demucs ready")
except Exception as e:
    print(f"✗ Demucs error: {e}")

try:
    from openunmix_separator import OpenUnmixSeparator
    openunmix_sep = OpenUnmixSeparator()
    print("✓ OpenUnmix ready")
except Exception as e:
    print(f"✗ OpenUnmix error: {e}")

print("All tests complete!")