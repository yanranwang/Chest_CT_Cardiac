#!/usr/bin/env python3
"""
测试心脏功能训练代码

这个脚本用于快速验证训练代码是否能正确运行，不会进行完整训练。
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np

def test_model_creation():
    """测试模型创建"""
    print("=== 测试模型创建 ===")
    try:
        from merlin.models.cardiac_regression import CardiacFunctionModel
        model = CardiacFunctionModel()
        print("✅ 模型创建成功")
        
        # 测试模型前向传播
        dummy_input = torch.randn(1, 1, 16, 224, 224)
        lvef_pred, as_pred = model(dummy_input)
        print(f"✅ 模型前向传播成功: LVEF shape {lvef_pred.shape}, AS shape {as_pred.shape}")
        return True
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False


def test_data_loading():
    """测试数据加载"""
    print("\n=== 测试数据加载 ===")
    try:
        from merlin.training.cardiac_trainer import load_and_validate_csv_data
        
        # 测试配置
        config = {
            'csv_path': '../filtered_echo_chestCT_data_filtered_chest_data.csv',
            'required_columns': ['basename', 'folder'],
            'cardiac_metric_columns': [],  # 空列表，使用模拟数据
            'metadata_columns': ['patient_id'],
            'remove_missing_files': True,
            'remove_duplicates': True,
            'check_file_exists': False
        }
        
        df, cardiac_metric_columns = load_and_validate_csv_data(config)
        print(f"✅ CSV数据加载成功: {len(df)} 行数据")
        return True
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False


def test_dataset_creation():
    """测试数据集创建"""
    print("\n=== 测试数据集创建 ===")
    try:
        from merlin.training.cardiac_trainer import CardiacDataset
        
        # 创建虚拟数据列表
        data_list = [
            {
                'image': f'dummy_image_{i}.nii.gz',
                'cardiac_metrics': None,  # 将使用模拟数据
                'patient_id': f'PATIENT_{i:03d}',
                'basename': f'dummy_{i}',
                'folder': f'folder_{i%5}',
                'metadata': {}
            }
            for i in range(10)
        ]
        
        dataset = CardiacDataset(data_list)
        print(f"✅ 数据集创建成功: {len(dataset)} 个样本")
        
        # 测试获取样本
        sample = dataset[0]
        print(f"✅ 样本获取成功: cardiac_metrics shape {sample['cardiac_metrics'].shape}")
        return True
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return False


def test_loss_function():
    """测试损失函数"""
    print("\n=== 测试损失函数 ===")
    try:
        from merlin.models.cardiac_regression import CardiacLoss
        
        criterion = CardiacLoss()
        
        # 创建虚拟预测和目标
        lvef_pred = torch.randn(4, 1)
        as_pred = torch.randn(4, 1)
        lvef_target = torch.randn(4)
        as_target = torch.randint(0, 2, (4,)).float()
        
        loss_dict = criterion(lvef_pred, as_pred, lvef_target, as_target)
        print(f"✅ 损失计算成功: 总损失 {loss_dict['total_loss']:.4f}")
        return True
    except Exception as e:
        print(f"❌ 损失函数测试失败: {e}")
        return False


def test_trainer_initialization():
    """测试训练器初始化"""
    print("\n=== 测试训练器初始化 ===")
    try:
        from merlin.training.cardiac_trainer import CardiacTrainer
        
        config = {
            'output_dir': 'test_outputs',
            'pretrained_model_path': None,
            'num_cardiac_metrics': 2,
            'epochs': 1,
            'batch_size': 2,
            'learning_rate': 1e-4,
            'device': 'cpu',  # 强制使用CPU避免GPU问题
            'use_tensorboard': False,  # 避免tensorboard问题
            'regression_weight': 1.0,
            'classification_weight': 1.0
        }
        
        trainer = CardiacTrainer(config)
        print("✅ 训练器初始化成功")
        
        # 清理测试输出目录
        import shutil
        if os.path.exists('test_outputs'):
            shutil.rmtree('test_outputs')
        
        return True
    except Exception as e:
        print(f"❌ 训练器初始化失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("开始测试心脏功能训练代码...")
    
    tests = [
        test_model_creation,
        test_data_loading,
        test_dataset_creation,
        test_loss_function,
        test_trainer_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！训练代码已准备就绪。")
        print("\n运行以下命令开始训练:")
        print("cd examples")
        print("python cardiac_training_example.py --epochs 2 --batch_size 2")
    else:
        print("⚠️  部分测试失败，请检查错误信息。")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 