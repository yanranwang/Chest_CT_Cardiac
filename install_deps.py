#!/usr/bin/env python3
"""
Training dependency installation script

This script automatically checks and installs required dependencies for cardiac training
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd):
    """Run system command"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_pip():
    """Check if pip is available"""
    success, stdout, stderr = run_command("pip --version")
    if success:
        print(f"✅ pip available")
        return True
    else:
        print("❌ pip not available")
        return False


def install_requirements():
    """Install dependencies from requirements.txt"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt file not found")
        return False
    
    print(f"📦 Installing dependencies...")
    print(f"Using file: {requirements_file}")
    
    # Upgrade pip first
    print("🔄 Upgrading pip...")
    success, stdout, stderr = run_command("pip install --upgrade pip")
    if not success:
        print("⚠️  pip upgrade failed, continuing...")
    
    # Install dependencies
    cmd = f"pip install -r {requirements_file}"
    success, stdout, stderr = run_command(cmd)
    
    if success:
        print("✅ Dependencies installed successfully")
        return True
    else:
        print("❌ Dependency installation failed")
        print(f"Error: {stderr}")
        return False


def install_torch_with_cuda():
    """Install PyTorch with CUDA support"""
    print("🔄 Checking CUDA support...")
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ PyTorch CUDA already available")
            return True
        else:
            print("⚠️  Current PyTorch doesn't support CUDA")
    except ImportError:
        print("⚠️  PyTorch not installed")
    
    # Ask whether to install CUDA version
    response = input("Install PyTorch with CUDA support? [y/N]: ").strip().lower()
    if response in ['y', 'yes']:
        print("🔄 Installing PyTorch with CUDA...")
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        success, stdout, stderr = run_command(cmd)
        if success:
            print("✅ CUDA PyTorch installed successfully")
            return True
        else:
            print("❌ CUDA PyTorch installation failed")
            print(f"Error: {stderr}")
            return False
    else:
        print("Using CPU version of PyTorch")
        return True


def verify_installation():
    """Verify installation"""
    print("\n🔍 Verifying installation...")
    
    # Check key packages
    packages = [
        'torch', 'torchvision', 'monai', 'tqdm', 
        'tensorboard', 'numpy', 'pandas', 'scikit-learn'
    ]
    
    all_success = True
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            all_success = False
    
    if all_success:
        print("\n🎉 All dependencies verified!")
        
        # Check CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
                print(f"✅ GPU count: {torch.cuda.device_count()}")
            else:
                print("⚠️  CUDA not available, will use CPU training")
        except:
            pass
        
        return True
    else:
        print("\n❌ Some dependencies failed verification")
        return False


def main():
    """Main function"""
    print("=" * 80)
    print("🔧 Merlin Cardiac Training Dependency Installer")
    print("=" * 80)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check pip
    if not check_pip():
        print("Please install pip first")
        sys.exit(1)
    
    # Install dependencies
    if not install_requirements():
        print("❌ Dependency installation failed")
        sys.exit(1)
    
    # Install CUDA support (optional)
    install_torch_with_cuda()
    
    # Verify installation
    if verify_installation():
        print("\n" + "=" * 80)
        print("🎉 Installation complete!")
        print("=" * 80)
        print("📚 Next steps:")
        print("   1. Run training example:")
        print("      cd examples")
        print("      python cardiac_training_example.py --epochs 5 --batch_size 2")
        print("   2. Monitor training progress:")
        print("      Progress bars and real-time statistics will be shown during training")
        print("   3. Monitor training:")
        print("      tensorboard --logdir outputs/cardiac_training/tensorboard")
        print("=" * 80)
    else:
        print("\n❌ Installation verification failed, please check error messages")
        sys.exit(1)


if __name__ == "__main__":
    main() 