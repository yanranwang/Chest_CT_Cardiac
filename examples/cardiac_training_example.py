#!/usr/bin/env python3
"""
心脏功能回归训练示例

该脚本展示了如何使用Merlin进行完整的心脏功能回归训练：
1. 从CSV文件加载心脏功能数据
2. 配置训练参数
3. 创建数据加载器
4. 训练心脏功能预测模型
5. 保存训练结果和模型

使用方法:
    python cardiac_training_example.py
    
或自定义配置:
    python cardiac_training_example.py --config my_config.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from merlin.training.cardiac_trainer import CardiacTrainer, create_data_loaders


def create_default_config():
    """创建默认训练配置"""
    config = {
        'output_dir': 'outputs/cardiac_training',
        'pretrained_model_path': '/dataNAS/people/joycewyr/Merlin/merlin/models/checkpoints/i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt',  # 使用Merlin预训练模型权重
        'num_cardiac_metrics': 2,  # LVEF回归 + AS分类
        'epochs': 100,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'loss_function': 'mse',
        'scheduler': {
            'type': 'cosine',
            'eta_min': 1e-6
        },
        'freeze_encoder': True,  # 重要：冻结预训练编码器进行微调
        'grad_clip': 1.0,
        'device': 'cuda',
        'seed': 42,
        'num_workers': 0,
        'log_interval': 5,  # 更频繁的日志输出（每5个batch）
        'save_interval': 5,  # 更频繁的模型保存（每5个epoch）
        'use_tensorboard': True,
        'drop_last': True,  # 训练时必须设置为True，避免BatchNorm错误
        
        # 进度监控增强配置
        'progress_bar': True,  # 启用进度条
        'show_gpu_memory': True,  # 显示GPU内存使用
        'show_eta': True,  # 显示预估剩余时间
        'detailed_metrics': True,  # 显示详细的评估指标
        
        # 数据分割配置
        'train_val_split': 0.8,
        'split_method': 'random',  # 'random', 'sequential', 'patient_based'
        
        # CSV数据配置
        'csv_path': '/dataNAS/people/joycewyr/Merlin/filtered_echo_chestCT_data_filtered_chest_data.csv',
        'required_columns': ['basename', 'folder'],
        'cardiac_metric_columns': [],  # 设置为CSV中包含心脏功能指标的列名列表
        'metadata_columns': ['patient_id'],  # 要保存的额外元数据列
        
        # 文件路径配置
        'base_path': '/dataNAS/data/ct_data/ct_scans',
        'image_path_template': '{base_path}/stanford_{folder}/{basename}.nii.gz',
        'check_file_exists': False,  # 设置为True以检查文件是否存在
        
        # 数据清理配置
        'remove_missing_files': True,
        'remove_duplicates': True,
        
        # 快速数据加载器配置
        'use_fast_loader': False,  # 设置为True以启用快速数据加载器
        'preprocessed_data_dir': 'outputs/preprocessed_data',  # 预处理数据目录
        'preprocess_batch_size': 16,  # 预处理批次大小
        'cache_config': {
            'enable_cache': True,      # 启用内存缓存
            'cache_size': 1000,        # 缓存大小
            'preload_train_data': False,  # 是否预加载训练数据到内存
            'preload_val_data': False,    # 是否预加载验证数据到内存
        }
    }
    return config


def load_config_from_file(config_path):
    """从JSON文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def save_config(config, output_path):
    """保存配置到JSON文件"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def print_training_info(config):
    """打印训练配置信息"""
    print("=" * 80)
    print("🔧 训练配置信息")
    print("=" * 80)
    print(f"📁 输出目录: {config['output_dir']}")
    print(f"📊 CSV文件: {config['csv_path']}")
    print(f"🏥 数据路径: {config['base_path']}")
    print(f"🎯 训练轮数: {config['epochs']}")
    print(f"📦 批量大小: {config['batch_size']}")
    print(f"🎓 学习率: {config['learning_rate']}")
    print(f"🔧 优化器: {config['optimizer']}")
    print(f"💾 日志间隔: {config['log_interval']} batches")
    print(f"💾 保存间隔: {config['save_interval']} epochs")
    print(f"🖥️  设备: {config['device']}")
    print(f"📈 TensorBoard: {'✅' if config['use_tensorboard'] else '❌'}")
    print(f"🎯 数据分割: {config['train_val_split']:.1%} 训练 / {1-config['train_val_split']:.1%} 验证")
    print("=" * 80)


def check_dependencies():
    """检查训练所需的依赖"""
    print("🔍 检查训练依赖...")
    
    missing_deps = []
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用, {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️  CUDA 不可用，将使用CPU训练")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import tqdm
        print(f"✅ tqdm (进度条)")
    except ImportError:
        missing_deps.append("tqdm")
        print("❌ tqdm 未安装，进度条将不可用")
    
    try:
        import tensorboard
        print(f"✅ TensorBoard")
    except ImportError:
        missing_deps.append("tensorboard")
        print("❌ TensorBoard 未安装，训练可视化将不可用")
    
    try:
        import monai
        print(f"✅ MONAI")
    except ImportError:
        missing_deps.append("monai")
    
    if missing_deps:
        print(f"\n⚠️  缺少依赖: {', '.join(missing_deps)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    print("✅ 所有依赖已安装")
    return True


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='心脏功能回归训练示例')
    parser.add_argument('--config', type=str, help='配置文件路径 (JSON格式)')
    parser.add_argument('--output_dir', type=str, help='输出目录路径')
    parser.add_argument('--csv_path', type=str, help='CSV数据文件路径')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批量大小')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    parser.add_argument('--log_interval', type=int, help='日志输出间隔')
    parser.add_argument('--save_interval', type=int, help='模型保存间隔')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'], help='训练设备')
    parser.add_argument('--use_pretrained', type=bool, default=True, help='是否使用预训练权重')
    parser.add_argument('--use_fast_loader', action='store_true', help='使用快速数据加载器')
    parser.add_argument('--preprocessed_data_dir', type=str, help='预处理数据目录')
    parser.add_argument('--preprocess_batch_size', type=int, help='预处理批次大小')
    
    args = parser.parse_args()
    
    # 检查依赖
    if not check_dependencies():
        print("❌ 依赖检查失败，请先安装缺少的依赖包")
        return
    
    # 加载配置
    if args.config:
        print(f"📋 从文件加载配置: {args.config}")
        config = load_config_from_file(args.config)
    else:
        print("📋 使用默认配置")
        config = create_default_config()
    
            # 验证和处理Merlin预训练权重
        if args.use_pretrained and config.get('pretrained_model_path'):
            print("\n🔍 检查Merlin预训练权重...")
            pretrained_path = config['pretrained_model_path']
            
            # 如果权重文件不存在，尝试自动下载
            if not os.path.exists(pretrained_path):
                print(f"❌ 预训练权重文件不存在: {pretrained_path}")
                print("🔄 尝试自动下载Merlin预训练权重...")
                
                try:
                    # 使用Merlin内置的权重下载功能
                    from merlin import Merlin
                    merlin_model = Merlin()  # 这会自动下载权重
                    
                    # 获取实际的权重路径
                    actual_checkpoint_path = os.path.join(
                        merlin_model.current_path, 
                        'checkpoints', 
                        merlin_model.checkpoint_name
                    )
                    
                    if os.path.exists(actual_checkpoint_path):
                        config['pretrained_model_path'] = actual_checkpoint_path
                        print(f"✅ 成功下载并设置预训练权重: {actual_checkpoint_path}")
                    else:
                        print("❌ 自动下载失败，将使用随机初始化权重")
                        config['pretrained_model_path'] = None
                        
                except Exception as e:
                    print(f"❌ 自动下载权重失败: {e}")
                    print("将使用随机初始化权重继续训练")
                    config['pretrained_model_path'] = None
            else:
                print(f"✅ 找到预训练权重文件: {pretrained_path}")
        
        # 检查是否使用快速数据加载器
        use_fast_loader = config.get('use_fast_loader', False)
        if use_fast_loader:
            print("\n🚀 使用快速数据加载器模式")
            preprocessed_data_dir = config.get('preprocessed_data_dir')
            if not preprocessed_data_dir:
                print("❌ 使用快速数据加载器需要设置 preprocessed_data_dir")
                print("请先运行数据预处理脚本:")
                print("  python -m merlin.training.data_preprocessor --config config.json")
                return
            
            # 检查预处理数据文件
            hdf5_path = Path(preprocessed_data_dir) / 'preprocessed_data.h5'
            metadata_path = Path(preprocessed_data_dir) / 'data_metadata.json'
            
            if not hdf5_path.exists():
                print(f"❌ 预处理数据文件不存在: {hdf5_path}")
                print("请先运行数据预处理脚本:")
                print("  python -m merlin.training.data_preprocessor --config config.json")
                return
            
            if not metadata_path.exists():
                print(f"❌ 元数据文件不存在: {metadata_path}")
                print("请先运行数据预处理脚本:")
                print("  python -m merlin.training.data_preprocessor --config config.json")
                return
            
            print(f"✅ 找到预处理数据: {hdf5_path}")
            print(f"✅ 找到元数据文件: {metadata_path}")
        else:
            print("\n📁 使用标准数据加载器模式")
    
    # 命令行参数覆盖配置
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.csv_path:
        config['csv_path'] = args.csv_path
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.log_interval:
        config['log_interval'] = args.log_interval
    if args.save_interval:
        config['save_interval'] = args.save_interval
    if args.device:
        config['device'] = args.device
    if args.use_fast_loader:
        config['use_fast_loader'] = True
    if args.preprocessed_data_dir:
        config['preprocessed_data_dir'] = args.preprocessed_data_dir
    if args.preprocess_batch_size:
        config['preprocess_batch_size'] = args.preprocess_batch_size
    
    # 打印配置信息
    print_training_info(config)
    
    # 保存配置
    output_dir = Path(config['output_dir'])
    config_save_path = output_dir / 'config.json'
    save_config(config, config_save_path)
    print(f"📁 配置已保存到: {config_save_path}")
    
    try:
        # 创建数据加载器
        print("\n📂 创建数据加载器...")
        use_fast_loader = config.get('use_fast_loader', False)
        
        if use_fast_loader:
            # 使用快速数据加载器
            from merlin.training.fast_dataloader import create_fast_data_loaders
            train_loader, val_loader = create_fast_data_loaders(config)
            print(f"✅ 使用快速数据加载器 - 训练集: {len(train_loader.dataset)}, 验证集: {len(val_loader.dataset)}")
        else:
            # 使用标准数据加载器
            train_loader, val_loader = create_data_loaders(config)
            print(f"✅ 使用标准数据加载器 - 训练集: {len(train_loader.dataset)}, 验证集: {len(val_loader.dataset)}")
        
        # 创建训练器
        print("\n🤖 初始化训练器...")
        trainer = CardiacTrainer(config)
        
        # 开始训练
        print("\n🚀 开始训练...")
        trainer.train(train_loader, val_loader)
        
        # 训练完成后的信息
        print("\n🎉 训练完成！")
        print("=" * 80)
        print("📁 输出文件:")
        print(f"   🏆 最佳模型: {config['output_dir']}/best_model.pth")
        print(f"   📊 训练日志: {config['output_dir']}/training.log")
        print(f"   ⚙️  配置文件: {config['output_dir']}/config.json")
        print(f"   📈 TensorBoard: {config['output_dir']}/tensorboard")
        print("\n💡 下一步:")
        print("   1. 查看训练日志了解详细信息")
        print("   2. 使用TensorBoard可视化训练过程:")
        print(f"      tensorboard --logdir {config['output_dir']}/tensorboard")
        print("   3. 使用训练好的模型进行预测")
        print("   4. 加速训练提示:")
        print("      - 预处理数据以加速后续训练:")
        print("        python -m merlin.training.data_preprocessor --config config.json")
        print("      - 使用快速数据加载器:")
        print("        python cardiac_training_example.py --use_fast_loader --preprocessed_data_dir outputs/preprocessed_data")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        print("已保存的检查点可用于恢复训练")
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        print("\n🔍 故障排除建议:")
        print("   1. 检查CSV文件路径是否正确")
        print("   2. 检查数据目录路径是否存在")
        print("   3. 检查GPU内存是否足够（可尝试减小batch_size）")
        print("   4. 检查磁盘空间是否充足")
        print("   5. 查看完整错误信息:")
        raise


if __name__ == '__main__':
    main() 