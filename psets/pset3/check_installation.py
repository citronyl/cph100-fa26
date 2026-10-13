"""Validate imports without contacting external services."""
import importlib

for name in ("numpy", "pandas", "sklearn", "matplotlib", "scipy", "groq"):
    module = importlib.import_module(name)
    print(f"{name}: OK ({getattr(module, '__version__', 'installed')})")
print("Setup succeeded. Download diabetic_data.csv as described in README.md.")
