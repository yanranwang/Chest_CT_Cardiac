#!/usr/bin/env python3
"""
训练依赖安装脚本

该脚本会自动检查和安装心脏功能训练所需的依赖包
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd):
    """运行系统命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ 是必需的")
        print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True


def check_pip():
    """检查pip是否可用"""
    success, stdout, stderr = run_command("pip --version")
    if success:
        print(f"✅ pip可用")
        return True
    else:
        print("❌ pip不可用")
        return False


def install_requirements():
    """安装requirements.txt中的依赖"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt文件不存在")
        return False
    
    print(f"📦 安装依赖包...")
    print(f"使用文件: {requirements_file}")
    
    # 先升级pip
    print("🔄 升级pip...")
    success, stdout, stderr = run_command("pip install --upgrade pip")
    if not success:
        print("⚠️  pip升级失败，继续安装...")
    
    # 安装依赖
    cmd = f"pip install -r {requirements_file}"
    success, stdout, stderr = run_command(cmd)
    
    if success:
        print("✅ 依赖安装成功")
        return True
    else:
        print("❌ 依赖安装失败")
        print(f"错误信息: {stderr}")
        return False


def install_torch_with_cuda():
    """安装支持CUDA的PyTorch"""
    print("🔄 检查CUDA支持...")
    
    # 检查CUDA是否可用
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ PyTorch CUDA已可用")
            return True
        else:
            print("⚠️  当前PyTorch不支持CUDA")
    except ImportError:
        print("⚠️  PyTorch未安装")
    
    # 询问是否安装CUDA版本
    response = input("是否安装支持CUDA的PyTorch？[y/N]: ").strip().lower()
    if response in ['y', 'yes']:
        print("🔄 安装支持CUDA的PyTorch...")
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        success, stdout, stderr = run_command(cmd)
        if success:
            print("✅ CUDA PyTorch安装成功")
            return True
        else:
            print("❌ CUDA PyTorch安装失败")
            print(f"错误信息: {stderr}")
            return False
    else:
        print("使用CPU版本的PyTorch")
        return True


def verify_installation():
    """验证安装"""
    print("\n🔍 验证安装...")
    
    # 检查关键包
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
        print("\n🎉 所有依赖验证成功！")
        
        # 检查CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
                print(f"✅ GPU数量: {torch.cuda.device_count()}")
            else:
                print("⚠️  CUDA不可用，将使用CPU训练")
        except:
            pass
        
        return True
    else:
        print("\n❌ 部分依赖验证失败")
        return False


def main():
    """主函数"""
    print("=" * 80)
    print("🔧 Merlin心脏功能训练依赖安装器")
    print("=" * 80)
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查pip
    if not check_pip():
        print("请先安装pip")
        sys.exit(1)
    
    # 安装依赖
    if not install_requirements():
        print("❌ 依赖安装失败")
        sys.exit(1)
    
    # 安装CUDA支持（可选）
    install_torch_with_cuda()
    
    # 验证安装
    if verify_installation():
        print("\n" + "=" * 80)
        print("🎉 安装完成！")
        print("=" * 80)
        print("📚 下一步:")
        print("   1. 运行训练示例:")
        print("      cd examples")
        print("      python cardiac_training_example.py --epochs 5 --batch_size 2")
        print("   2. 查看训练进度:")
        print("      训练过程中会显示进度条和实时统计信息")
        print("   3. 监控训练:")
        print("      tensorboard --logdir outputs/cardiac_training/tensorboard")
        print("=" * 80)
    else:
        print("\n❌ 安装验证失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main() 