"""Validate imports without contacting external services."""
import importlib

for name in ("awscli", "numpy", "pandas", "sklearn", "matplotlib", "scipy", "mne", "braindecode", "ipykernel", "nbstripout"):
    module = importlib.import_module(name)
    print(f"{name}: OK ({getattr(module, '__version__', 'installed')})")
print("Setup succeeded.")
