#!/usr/bin/env python3
"""
心脏功能预测演示脚本

该脚本展示了如何使用Merlin进行心脏功能回归训练和预测：
1. 加载Merlin预训练模型
2. 进行心脏功能回归训练
3. 执行心脏功能预测
4. 生成预测报告

使用方法:
    python cardiac_demo.py --mode train    # 训练模式
    python cardiac_demo.py --mode inference --model_path outputs/cardiac_training/best_model.pth
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from merlin.models.cardiac_regression import CardiacFunctionModel, CardiacMetricsCalculator
from merlin.training.cardiac_trainer import CardiacTrainer, create_data_loaders
from merlin.inference.cardiac_inference import CardiacInference
from merlin.data import download_sample_data, DataLoader


def create_demo_config():
    """创建演示训练配置"""
    config = {
        'output_dir': 'outputs/cardiac_training',
        'pretrained_model_path': None,  # 将会自动下载Merlin预训练权重
        'num_cardiac_metrics': 10,
        'epochs': 20,  # 演示用较少epoch
        'batch_size': 2,  # 演示用较小batch size
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'loss_function': 'mse',
        'scheduler': {
            'type': 'cosine',
            'eta_min': 1e-6
        },
        'freeze_encoder': True,  # 冻结预训练编码器
        'grad_clip': 1.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'num_workers': 2,
        'log_interval': 5,
        'save_interval': 5,
        'use_tensorboard': True,
        'train_val_split': 0.8,
        'num_samples': 50  # 演示用较少样本
    }
    return config


def generate_synthetic_cardiac_data(num_samples=50):
    """生成合成的心脏功能数据用于演示"""
    print("生成合成心脏功能数据...")
    
    data_list = []
    
    # 模拟不同类型的心脏状况
    conditions = ['normal', 'mild_dysfunction', 'moderate_dysfunction', 'severe_dysfunction']
    
    for i in range(num_samples):
        condition = conditions[i % len(conditions)]
        
        # 根据心脏状况生成相应的功能指标
        if condition == 'normal':
            # 正常心脏功能
            ef = np.random.normal(60, 5)  # 射血分数
            sv = np.random.normal(70, 10)  # 每搏输出量
            co = np.random.normal(5.0, 0.5)  # 心输出量
            hrv = np.random.normal(30, 5)  # 心率变异性
            lvm = np.random.normal(150, 20)  # 左心室质量
            wt = np.random.normal(10, 1)  # 室壁厚度
            cv = np.random.normal(120, 15)  # 心室容积
            ci = np.random.normal(1.0, 0.1)  # 收缩性指数
            df = np.random.normal(1.0, 0.1)  # 舒张功能
            vf = np.random.normal(1.0, 0.1)  # 瓣膜功能
            
        elif condition == 'mild_dysfunction':
            # 轻度功能障碍
            ef = np.random.normal(50, 5)
            sv = np.random.normal(60, 10)
            co = np.random.normal(4.0, 0.5)
            hrv = np.random.normal(25, 5)
            lvm = np.random.normal(170, 20)
            wt = np.random.normal(12, 1)
            cv = np.random.normal(140, 15)
            ci = np.random.normal(0.8, 0.1)
            df = np.random.normal(0.8, 0.1)
            vf = np.random.normal(0.9, 0.1)
            
        elif condition == 'moderate_dysfunction':
            # 中度功能障碍
            ef = np.random.normal(40, 5)
            sv = np.random.normal(50, 10)
            co = np.random.normal(3.5, 0.5)
            hrv = np.random.normal(20, 5)
            lvm = np.random.normal(200, 20)
            wt = np.random.normal(15, 1)
            cv = np.random.normal(160, 15)
            ci = np.random.normal(0.6, 0.1)
            df = np.random.normal(0.6, 0.1)
            vf = np.random.normal(0.7, 0.1)
            
        else:  # severe_dysfunction
            # 重度功能障碍
            ef = np.random.normal(30, 5)
            sv = np.random.normal(40, 10)
            co = np.random.normal(2.5, 0.5)
            hrv = np.random.normal(15, 5)
            lvm = np.random.normal(250, 20)
            wt = np.random.normal(18, 1)
            cv = np.random.normal(180, 15)
            ci = np.random.normal(0.4, 0.1)
            df = np.random.normal(0.4, 0.1)
            vf = np.random.normal(0.5, 0.1)
        
        # 转换为标准化值（模型输入格式）
        cardiac_metrics = np.array([ef, sv, co, hrv, lvm, wt, cv, ci, df, vf])
        
        # 标准化到[-1, 1]范围
        cardiac_metrics_normalized = (cardiac_metrics - np.array([60, 70, 5, 30, 150, 10, 120, 1, 1, 1])) / \
                                    np.array([10, 15, 1, 10, 30, 2, 25, 0.2, 0.3, 0.2])
        
        data_list.append({
            'image': f'synthetic_ct_{i:03d}.nii.gz',  # 模拟CT图像文件名
            'cardiac_metrics': cardiac_metrics_normalized.astype(np.float32),
            'patient_id': f'DEMO_{i:03d}',
            'condition': condition,
            'raw_metrics': cardiac_metrics
        })
    
    print(f"生成了 {len(data_list)} 个合成数据样本")
    return data_list


def train_cardiac_model():
    """训练心脏功能预测模型"""
    print("开始训练心脏功能预测模型...")
    
    # 创建配置
    config = create_demo_config()
    
    # 生成合成数据
    synthetic_data = generate_synthetic_cardiac_data(config['num_samples'])
    
    # 创建数据加载器
    print("创建数据加载器...")
    
    # 分割训练和验证数据
    split_idx = int(len(synthetic_data) * config['train_val_split'])
    train_data = synthetic_data[:split_idx]
    val_data = synthetic_data[split_idx:]
    
    # 使用Merlin的数据加载器结构
    from merlin.training.cardiac_trainer import CardiacDataset
    from torch.utils.data import DataLoader
    
    train_dataset = CardiacDataset(train_data)
    val_dataset = CardiacDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # 创建训练器
    trainer = CardiacTrainer(config)
    
    # 开始训练
    trainer.train(train_loader, val_loader)
    
    print("训练完成！")
    print(f"最佳模型保存在: {config['output_dir']}/best_model.pth")
    
    return config['output_dir']


def test_cardiac_inference(model_path):
    """测试心脏功能推理"""
    print(f"使用模型进行心脏功能推理: {model_path}")
    
    # 生成一些测试数据
    test_data = generate_synthetic_cardiac_data(5)
    
    try:
        # 创建推理器（注意：这里需要实际的CT图像文件）
        predictor = CardiacInference(model_path)
        
        print("\n心脏功能预测演示:")
        print("=" * 60)
        
        for i, sample in enumerate(test_data[:3]):  # 只测试前3个样本
            print(f"\n患者 {sample['patient_id']} (模拟{sample['condition']}):")
            print("-" * 40)
            
            # 真实的心脏功能指标
            real_metrics = sample['raw_metrics']
            metric_names = CardiacMetricsCalculator.get_metric_names()
            
            print("真实心脏功能指标:")
            for j, (name, value) in enumerate(zip(metric_names, real_metrics)):
                print(f"  {name:25}: {value:8.2f}")
            
            # 注意：由于没有真实的CT图像文件，这里会报错
            # 在实际使用中，需要提供真实的.nii.gz文件路径
            print("\n注意：此演示需要真实的CT图像文件才能完成预测")
            
        print("\n" + "=" * 60)
        print("演示完成！")
        
    except Exception as e:
        print(f"推理演示失败: {str(e)}")
        print("这是正常的，因为演示中没有提供真实的CT图像文件")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='心脏功能预测演示')
    parser.add_argument('--mode', choices=['train', 'inference', 'all'], 
                       default='all', help='运行模式')
    parser.add_argument('--model_path', type=str, 
                       default='outputs/cardiac_training/best_model.pth',
                       help='模型权重路径（推理模式）')
    parser.add_argument('--image_path', type=str,
                       help='CT图像路径（推理模式）')
    
    args = parser.parse_args()
    
    print("Merlin心脏功能预测演示")
    print("=" * 60)
    print(f"运行模式: {args.mode}")
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("")
    
    if args.mode in ['train', 'all']:
        # 训练模式
        output_dir = train_cardiac_model()
        model_path = os.path.join(output_dir, 'best_model.pth')
    else:
        model_path = args.model_path
    
    if args.mode in ['inference', 'all']:
        # 推理模式
        if os.path.exists(model_path):
            test_cardiac_inference(model_path)
        else:
            print(f"模型文件不存在: {model_path}")
            print("请先运行训练模式或提供正确的模型路径")
    
    print("\n演示脚本执行完成！")
    print("\n使用说明:")
    print("1. 训练: python cardiac_demo.py --mode train")
    print("2. 推理: python cardiac_demo.py --mode inference --model_path path/to/model.pth")
    print("3. 完整流程: python cardiac_demo.py --mode all")


if __name__ == "__main__":
    main() 