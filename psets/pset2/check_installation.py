"""
Simple installation check for CPH 100A Problem Set 2.
"""

import sys

def check_basic_packages():
    """Check if basic packages are installed."""
    print("Checking package installations...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check required packages
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('medmnist', 'MedMNIST'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('tqdm', 'tqdm')
    ]
    
    missing_packages = []
    
    for package, name in packages:
        try:
            module = __import__(package)
            print(f"✅ {name}: {getattr(module, '__version__', 'installed')}")
        except ImportError:
            print(f"❌ {name}: not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def test_pytorch():
    """Test basic PyTorch functionality."""
    print("\nTesting PyTorch...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Create a simple tensor
        x = torch.randn(2, 3)
        print(f"✅ Tensor creation: {x.shape}")
        
        # Create a simple layer
        layer = nn.Linear(3, 1)
        output = layer(x)
        print(f"✅ Neural network layer: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_medmnist():
    """Test MedMNIST data loading."""
    print("\nTesting MedMNIST...")
    
    try:
        from medmnist import PathMNIST
        
        # Try to create dataset (without downloading)
        dataset = PathMNIST(split='train', download=True, root='./data')
        print("✅ MedMNIST import successful")
        return True
        
    except Exception as e:
        print(f"❌ MedMNIST test failed: {e}")
        print("This is normal if you haven't downloaded the data yet.")
        return True  # Don't fail on this

def main():
    """Run all installation checks."""
    print("CPH 100A Problem Set 2 - Installation Check")
    print("=" * 50)
    
    all_good = True
    
    # Check packages
    all_good &= check_basic_packages()
    
    # Test PyTorch
    all_good &= test_pytorch()
    
    # Test MedMNIST
    test_medmnist()
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 Installation check passed!")
        print("\nNext steps:")
        print("1. python classification_main.py")
        print("2. python segmentation_main.py")
    else:
        print("❌ Installation issues found.")
        print("Please install missing packages and try again.")

if __name__ == "__main__":
    main()